"""
Promote the best fine-tuning experiment to Hugging Face, but only if it beats the model
that is there now.

1. Pick: the experiment with the highest mean dev macro F1 over its seeds in the
   results file of experiments.py, and that experiment's seed with the best dev macro F1.
   --experiment and --seed override the choice.
2. Build: reuse the saved model in output/.../models/<name>-seed<seed> (experiments.py
   --save), or else train it again with experiments.py --save.
3. Check: score the candidate and the model on Hugging Face on the same dev and test
   sets. The Hugging Face model is downloaded and scored, not looked up, so the check
   works after local experiments are deleted and also catches a repo whose weights and
   config don't belong together.
4. Upload: only when the candidate's dev macro F1 beats the Hugging Face model's by more
   than --min-gain (or the repo does not exist yet). For the tweet model, the decision
   thresholds (thresholds.py) and the int8 ONNX model (export_onnx.py) are then redone
   for the new weights, so they never belong to the old model.

Every run writes the comparison to output/<experiments dir>/promotion.json. Run from the
repository root, logged in to Hugging Face (`hf auth login`):

    python ModernFinTwitBERT/promote.py --dry-run          # tweets: see what would happen
    python ModernFinTwitBERT/promote.py                    # tweets: FinTwitRoBERTa-sentiment
    python ModernFinTwitBERT/promote.py --config experiments_wsb.json   # FinTwitRoBERTa-wsb-sentiment
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from experiments import labeled, test_metrics, upload
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from sentiment_card import DEFAULT_REPO
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from wsb import WSB_REPO, load_wsb_dev, load_wsb_test

from data import load_dev_data, load_test_data

METRIC = "dev/macro_f1"
SCRIPTS = Path(__file__).parent


def pick(results: pd.DataFrame, experiment: str | None, seed: int | None) -> tuple[str, int]:
    """The experiment with the best mean dev score and its best seed, unless given."""
    results = results.dropna(subset=[METRIC])
    if results.empty:
        sys.exit(f"No runs with {METRIC} in the results file; run experiments.py first")
    grouped = results.groupby("name")[METRIC].agg(["mean", "std", "count"])
    grouped = grouped.sort_values("mean", ascending=False)
    print(f"Experiments by mean {METRIC}:")
    print(grouped.head(5).round(3).to_string(), "\n")
    if experiment is None:
        experiment = grouped.index[0]
        if len(grouped) > 1:
            gap = grouped["mean"].iloc[0] - grouped["mean"].iloc[1]
            spread = grouped["std"].iloc[0]
            if pd.notna(spread) and gap < spread:
                print(
                    f"Note: {experiment} leads {grouped.index[1]} by {gap:.3f}, less than its "
                    f"seed std ({spread:.3f}); pass --experiment to choose on other grounds.\n"
                )
    elif experiment not in grouped.index:
        sys.exit(f"No runs of {experiment} in the results file")
    runs = results[results["name"] == experiment]
    if seed is None:
        seed = int(runs.loc[runs[METRIC].idxmax(), "seed"])
    return experiment, seed


def build(config: str, output_dir: Path, experiment: str, seed: int) -> Path:
    """The saved model of a run, trained again with --save when it is not on disk."""
    folder = output_dir / "models" / f"{experiment}-seed{seed}"
    if (folder / "config.json").exists():
        print(f"Using the saved model in {folder}")
        return folder
    print(f"{folder} does not exist: training {experiment} seed {seed} again with --save")
    command = [
        sys.executable,
        str(SCRIPTS / "experiments.py"),
        "--config",
        config,
        "--only",
        experiment,
        "--exact",
        "--seeds",
        str(seed),
        "--save",
    ]
    subprocess.run(command, check=True)
    return folder


def evaluate(model_id: str, splits: dict, max_length: int) -> dict:
    """Macro F1 and the other test metrics of a local or Hugging Face model, per split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    temperature = getattr(model.config, "calibration_temperature", None) or 1.0
    row = {"temperature": temperature}
    for split, dataset in splits.items():
        texts, logits = dataset["text"], []
        for start in range(0, len(texts), 32):
            inputs = tokenizer(
                texts[start : start + 32],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                logits.append(model(**inputs).logits.float().cpu())
        probs = (torch.cat(logits) / temperature).softmax(-1).numpy()
        row.update(test_metrics(probs, labeled(dataset), prefix=split))
    return row


def hub_exists(repo: str) -> bool:
    try:
        HfApi().model_info(repo)
        return True
    except RepositoryNotFoundError:
        return False


def run_script(script: str, *args: str):
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="experiments.json", help="Experiments file")
    parser.add_argument("--hub-repo", help="By default FinTwitRoBERTa-(wsb-)sentiment")
    parser.add_argument("--experiment", help="Promote this experiment instead of the best")
    parser.add_argument("--seed", type=int, help="Promote this seed instead of the best")
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help="Dev macro F1 the candidate must gain over the Hugging Face model (default 0)",
    )
    parser.add_argument("--force", action="store_true", help="Upload even if not better")
    parser.add_argument("--dry-run", action="store_true", help="Compare, but don't upload")
    parser.add_argument(
        "--public", action="store_true", help="Make a new repo public (default private)"
    )
    args = parser.parse_args()

    plan = json.loads(Path(args.config).read_text())
    config = json.loads(Path("config.json").read_text())
    wsb = plan.get("domain") == "wsb"
    output_dir = Path("output/experiments_wsb" if wsb else "output/experiments")
    repo = args.hub_repo or (WSB_REPO if wsb else DEFAULT_REPO)
    max_length = plan.get("max_length") or config["finetune"].get("max_length", 128)

    results_path = output_dir / "results.csv"
    if not results_path.exists():
        sys.exit(f"{results_path} does not exist; run experiments.py --config {args.config}")
    experiment, seed = pick(pd.read_csv(results_path), args.experiment, args.seed)
    print(f"Candidate: {experiment} seed {seed}\n")
    candidate = build(args.config, output_dir, experiment, seed)

    if wsb:
        splits = {"dev": load_wsb_dev(), "test": load_wsb_test(), "tweets": load_test_data()}
    else:
        splits = {"dev": load_dev_data(), "test": load_test_data()}
    scores = {"candidate": evaluate(str(candidate), splits, max_length)}
    exists = hub_exists(repo)
    if exists:
        scores["hub"] = evaluate(repo, splits, max_length)

    keys = [
        k
        for k in scores["candidate"]
        if k.endswith(("/macro_f1", "/precision_bullish", "/precision_bearish", "/ece"))
    ]
    table = pd.DataFrame({name: {k: row.get(k) for k in keys} for name, row in scores.items()})
    print(f"\nCandidate vs {repo}:" if exists else f"\n{repo} does not exist yet:")
    print(table.round(3).to_string(), "\n")

    gain = scores["candidate"][METRIC] - scores["hub"][METRIC] if exists else None
    if not exists:
        decision, reason = True, "the repo does not exist yet"
    elif gain > args.min_gain:
        decision, reason = True, f"dev macro F1 +{gain:.3f} (> {args.min_gain})"
    else:
        decision, reason = args.force, f"dev macro F1 {gain:+.3f} (not > {args.min_gain})"
        if args.force:
            reason += ", uploaded with --force"
    if args.dry_run:
        reason += " [dry run]"

    report = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": repo,
        "experiment": experiment,
        "seed": seed,
        "model": str(candidate),
        "uploaded": decision and not args.dry_run,
        "reason": reason,
        "scores": scores,
    }
    (output_dir / "promotion.json").write_text(json.dumps(report, indent=2))
    print(f"{'Upload' if decision else 'Keep the Hugging Face model'}: {reason}")
    if not decision or args.dry_run:
        return

    upload(candidate, repo, private=not args.public)
    if not wsb:
        # Thresholds and the ONNX export belong to the weights: redo them for the new ones
        run_script("thresholds.py", "--model", str(candidate), "--recompute", "--hub-repo", repo)
        run_script("export_onnx.py", "--model", str(candidate), "--hub-repo", repo)
    print(f"\nPromoted {experiment} seed {seed} to https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
