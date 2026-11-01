# FinTwitRoBERTa: Language Models for Financial Tweets

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Supported versions">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FinTwitRoBERTa--sentiment-yellow" alt="Hugging Face"></a>
</p>

FinTwitRoBERTa is a family of language models for financial Twitter (FinTwit). It reads the tickers, slang and shorthand of stock and crypto tweets and tells you whether a tweet is **bullish**, **bearish** or **neutral**. It is the successor of [FinTwitBERT](https://github.com/TimKoornstra/FinTwitBERT): on the same held-out tweets, the macro F1 goes from **0.565 to 0.891**.

## Introduction

Most of what is posted on financial Twitter is not an opinion: it is news, price alerts, questions and bot spam. A useful sentiment model has to recognise that, and only call a tweet bullish or bearish when its author actually expects a move or holds a position. The original FinTwitBERT-sentiment was trained on labels that often called factual tweets bullish or bearish, so on a real feed it called almost two out of three neutral tweets directional.

FinTwitRoBERTa was rebuilt with that use case in mind, running sentiment over a live FinTwit feed for market insights. Compared with FinTwitBERT, it changes:

- **Data:** one labeling guideline for all training data. All 13,206 bullish and bearish tweets of the kept sources were relabeled by hand (most became neutral), and 1,500 tweets from a 2024+ feed were added.
- **Evaluation:** a frozen test set and dev set of real feed tweets, kept out of all pretraining and training.
- **Base model:** chosen by benchmarking a dozen candidates (ModernBERT, DeBERTa-v3, Ettin, Twitter-RoBERTa and others).
- **Pretraining:** continued masked-language-model training on millions of cleaned financial tweets.
- **Outputs:** calibrated confidence scores, tuned decision thresholds and a 4x smaller int8 ONNX model.

## Table of Contents

- [Models](#models)
- [Results](#results)
- [Datasets](#datasets)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Models

All models are public on Hugging Face:

| Model | Task | Use it for |
| --- | --- | --- |
| [FinTwitRoBERTa-large](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-large) | Fill-mask (pre-trained) | Starting point for your own financial-tweet tasks |
| [FinTwitRoBERTa-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment) | Sentiment: neutral / bullish / bearish | Tweets and other short financial social media posts. Includes an int8 ONNX model |
| [FinTwitRoBERTa-wsb-sentiment](https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-wsb-sentiment) | Sentiment: neutral / bullish / bearish | r/wallstreetbets posts and comments |

## Results

### Tweets

Scored on the test set of 1,237 tweets from a real financial Twitter feed (702 neutral, 380 bullish, 155 bearish; see [Datasets](#evaluation-datasets)). ECE is the calibration error: lower means the confidence scores better match how often the model is right.

| Model | Macro F1 | Accuracy | Bullish precision | Bearish precision | Neutral recall | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| **FinTwitRoBERTa-sentiment** | **0.891** | **0.906** | **0.884** | **0.938** | **0.934** | **0.017** |
| [FinTwitBERT-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) | 0.565 | 0.575 | 0.559 | 0.354 | 0.350 | 0.251 |
| [ModernBERT financial news](https://huggingface.co/mrm8488/ModernBERT-base-ft-financial-news-sentiment-analysis) | 0.505 | 0.618 | 0.603 | 0.426 | 0.840 | 0.305 |
| [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | 0.482 | 0.564 | 0.521 | 0.319 | 0.715 | 0.231 |

The 95% bootstrap interval of FinTwitRoBERTa-sentiment's macro F1 is 0.871–0.910. With the [decision thresholds](#calibrated-scores-and-decision-thresholds) applied, its test precision is 0.95 for bullish and 0.99 for bearish calls, at a recall of 0.82 and 0.63.

### r/wallstreetbets

Scored on the same 227 held-out [WSBS](https://github.com/moonscape95/WSBS) posts that FinTwitBERT-wsb-sentiment was tested on:

| Model | Macro F1 | Accuracy |
| --- | --- | --- |
| **FinTwitRoBERTa-wsb-sentiment** | **0.886** | **0.894** |
| [FinTwitBERT-wsb-sentiment](https://huggingface.co/StephanAkkerman/FinTwitBERT-wsb-sentiment) | 0.821 | – |
| FinTwitRoBERTa-sentiment (no WSB training) | 0.657 | – |

With only 227 posts, differences of a few points are noise (95% interval 0.838–0.924).

### Choosing the base model

Every candidate was fine-tuned with the same recipe and scored on the tweet test set (mean macro F1 over 3 seeds). Twitter-RoBERTa-large won, and continuing its pretraining on financial tweets added another 0.033:

| Base model | Macro F1 |
| --- | --- |
| **FinTwitRoBERTa-large** (ours, Twitter-RoBERTa-large + financial tweets) | **0.840** |
| [cardiffnlp/twitter-roberta-large-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m) | 0.807 |
| [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) | 0.804 |
| [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | 0.761 |
| [BatuhanECB/FinModernBERT-large-DAPT](https://huggingface.co/BatuhanECB/FinModernBERT-large-DAPT) | 0.748 |
| [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | 0.742 |
| [jhu-clsp/ettin-encoder-400m](https://huggingface.co/jhu-clsp/ettin-encoder-400m) | 0.739 |
| [StephanAkkerman/FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT) (previous model) | 0.738 |
| [cardiffnlp/twitter-roberta-base-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2022-154m) | 0.734 |
| [clapAI/Fin-ModernBERT](https://huggingface.co/clapAI/Fin-ModernBERT) | 0.715 |
| [jhu-clsp/ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m) | 0.706 |
| [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 0.692 |

The released FinTwitRoBERTa-sentiment (0.891) is FinTwitRoBERTa-large with the best fine-tuning recipe from [the experiments](#experiments) and the extra feed tweets.

## Datasets

### Pre-training datasets

FinTwitRoBERTa-large was pre-trained on millions of cleaned financial tweets:

- [StephanAkkerman/financial-tweets](https://huggingface.co/datasets/StephanAkkerman/financial-tweets): a curated financial Twitter feed, shown 3 times per epoch to upweight it.
- [StephanAkkerman/crypto-stock-tweets](https://huggingface.co/datasets/StephanAkkerman/crypto-stock-tweets).

Near-duplicates (such as bot price alerts that only differ in numbers), one-word tweets and tweets that are mostly hashtags, cashtags, mentions or links were removed, and quoted tweets became examples of their own. The test and dev tweets were removed.

### Fine-tuning datasets

- [TimKoornstra/financial-tweets-sentiment](https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment): 24,373 human-labeled tweets after cleaning. Sources without a neutral class or with inconsistent labels are left out, and the bullish and bearish tweets of the other sources were relabeled with one labeling guideline ([`data/train/relabels_v1.csv`](data/train/relabels_v1.csv)). Most became neutral, because they state facts or news rather than an opinion.
- [`data/train/feed_v1.csv`](data/train/feed_v1.csv): 1,500 tweets from the 2024+ feed, picked where an earlier model was least sure and labeled with the same guideline.

The guideline in short: a tweet is bullish or bearish only when its author expresses an expectation or a position. Facts, news, questions and price alerts are neutral. For a quote tweet, only the new text counts.

FinTwitBERT's [synthetic dataset](https://huggingface.co/datasets/TimKoornstra/synthetic-financial-tweets-sentiment) is not used.

### Evaluation datasets

Both come from the same real feed as `financial-tweets` and are never used for pretraining or training:

- [`data/test/test_v0.csv`](data/test/test_v0.csv): 1,237 tweets first seen from August 2023 onwards (702 neutral, 380 bullish, 155 bearish). Only used to confirm results.
- [`data/test/dev_v0.csv`](data/test/dev_v0.csv): 749 tweets from 2024 onwards (541 neutral, 143 bullish, 65 bearish). Used to choose recipes, fit the calibration temperature and tune the decision thresholds.

## Model Details

- **FinTwitRoBERTa-large** continues the pretraining of [cardiffnlp/twitter-roberta-large-2022-154m](https://huggingface.co/cardiffnlp/twitter-roberta-large-2022-154m), a RoBERTa-large model (355M parameters) trained on 154M tweets up to the end of 2022. It ran 1 epoch of masked language modeling (15% masking, learning rate 5e-5, batch 256, max 128 tokens) on an A100 80GB. The tokenizer is unchanged. Links become `[URL]` and mentions `@USER` as plain text, not as added tokens. Perplexity on the 1,986 held-out tweets is 3.43, against 166 for the base model.
- **FinTwitRoBERTa-sentiment** fine-tunes FinTwitRoBERTa-large for 2 epochs (learning rate 2e-5, 10% warmup, layer-wise learning-rate decay 0.9, batch 32). This is the `llrd-0.9-feed` recipe in [`experiments.json`](experiments.json), chosen on the dev set out of a grid of learning rates, epochs, class weights, label smoothing and layer-wise decay.
  - The config also stores `calibration_temperature` (1.046) and `decision_thresholds` (bullish 0.80, bearish 0.86).
  - Labels are `NEUTRAL`, `BULLISH` and `BEARISH`.
- **FinTwitRoBERTa-wsb-sentiment** fine-tunes FinTwitRoBERTa-sentiment on 1,056 r/wallstreetbets posts from WSBS (`from-sentiment-lr2e-05` in [`experiments_wsb.json`](experiments_wsb.json), max 256 tokens, calibration temperature 2.011).

## Installation

```bash
# Clone this repository
git clone https://github.com/StephanAkkerman/FinTwitRoBERTa
cd FinTwitRoBERTa
# Install required packages
pip install -r requirements.txt
```

To only use the models, `pip install transformers torch` is enough. The ONNX model needs only `onnxruntime tokenizers huggingface_hub`.

## Usage

### Pre-trained model

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="StephanAkkerman/FinTwitRoBERTa-large")
print(pipe("Fed cuts rates, $SPY to all time <mask>."))
```

### Sentiment model

```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="StephanAkkerman/FinTwitRoBERTa-sentiment")
print(pipe("Nice 9% pre market move for $para, pump my calls Uncle Buffett 🤑"))
```

The model was trained with links replaced by `[URL]` and user mentions by `@USER`. Do the same for the best results: `preprocess_tweet()` in [`ModernFinTwitBERT/preprocess.py`](ModernFinTwitBERT/preprocess.py) does this, and also strips the quoted tweet from a quote tweet.

### Calibrated scores and decision thresholds

The pipeline's scores are not calibrated. For confidence scores you can act on, divide the logits by `config.calibration_temperature`. Then only call a tweet bullish or bearish when its probability reaches `config.decision_thresholds`. These are tuned for 90% precision on the dev set, so the model stays neutral when unsure:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "StephanAkkerman/FinTwitRoBERTa-sentiment"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

inputs = tokenizer(["$NVDA breaking out, loading calls"], return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
probs = (logits / model.config.calibration_temperature).softmax(-1)[0]

label = model.config.id2label[int(probs.argmax())].lower()
if label != "neutral" and probs.max() < model.config.decision_thresholds[label]:
    label = "neutral"
print(label, probs.tolist())
```

### Fast CPU inference (ONNX int8)

`onnx/model_int8.onnx` in the sentiment repo is 4x smaller (358 MB instead of 1.4 GB) and about 5x faster on CPU (19 ms instead of 97 ms per tweet) with the same F1. It runs on small devices such as a Raspberry Pi. [`ModernFinTwitBERT/onnx_predict.py`](ModernFinTwitBERT/onnx_predict.py) needs no torch or transformers, and applies the preprocessing, temperature and thresholds for you:

```python
from onnx_predict import OnnxSentiment

model = OnnxSentiment()  # downloads StephanAkkerman/FinTwitRoBERTa-sentiment
print(model.predict(["$NVDA breaking out, adding here"]))
# [{"label": "bullish", "probs": {"neutral": 0.04, "bullish": 0.95, "bearish": 0.01}}]
```

### r/wallstreetbets model

```python
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="StephanAkkerman/FinTwitRoBERTa-wsb-sentiment")
print(pipe("Come on TSLA. Fly you regarded bird!!"))
```

Use this one for Reddit posts and FinTwitRoBERTa-sentiment for tweets: on tweets, the WSB model calls too many neutral tweets directional (tweet test macro F1 0.735).

## Training

All scripts are in [`ModernFinTwitBERT/`](ModernFinTwitBERT/) and run from the repository root. The full pipeline is:

1. Pretrain on financial tweets (`model.py`, mode `pretrain`).
2. Compare base models (`benchmark.py`).
3. Run the fine-tuning experiments and pick a recipe on the dev set (`experiments.py`).
4. Tune the decision thresholds (`thresholds.py`) and export the ONNX model (`export_onnx.py`).
5. Upload the winner to Hugging Face if it beats the current model (`promote.py`, which also redoes step 4).

Training is configured in [`config.json`](config.json). Fine-tuning runs are logged to [Weights & Biases](https://wandb.ai): run `wandb login` once, or add a `wandb.env` file with `WANDB_API_KEY=your_wandb_api_key`. Without a login, results are only saved locally. Log in to Hugging Face with `hf auth login` before uploading.

### Pretraining

Set `"mode": "pretrain"` in `config.json` and run `python ModernFinTwitBERT/model.py`. This continues masked-language-model training on the sources in `pretrain.datasets` and saves to `output/<pretrain model_name>`. One epoch takes a few hours on an A100.

- Start with `python ModernFinTwitBERT/model.py --max-steps 50`: a short trial that saves nothing, to check speed and memory on a new GPU.
- On a rented GPU, [`scripts/pretrain_remote.sh`](scripts/pretrain_remote.sh) does everything. It installs the requirements (without flash-attn, which RoBERTa does not use), runs a 50-step trial, then the full run, and uploads the model with its card to `pretrain.hub_repo` (or `HF_REPO`). A new repo is created private. The default `hub_repo` is the released model, so change it before a new run unless you mean to replace it.
- The batch of 256 is sized for an 80 GB A100. On a 40 GB GPU, use 128 with `gradient_accumulation_steps` 2.
- `pretrain.datasets` lists the sources: Hugging Face ids or local `.csv`/`.jsonl`/`.parquet` files, each with an optional `text_column` and a `repeat` to upweight it. Put curated sources first, because near-duplicates are kept in the first source listed.
- Cleaned tweets are cached in `datasets/pretrain_clean`, and cleaned again when a source, the cleaning code or the test set changes. Checkpoints go to `output/checkpoints/<model_name>`, and running the same command again resumes from the last one.
- At the end, the run measures the perplexity of the model and its base model on the held-out test and dev tweets, and writes a Hugging Face model card (`README.md`) and `pretrain_stats.json` next to the model. `python ModernFinTwitBERT/pretrain_card.py` rewrites the card later, e.g. to add benchmark results.

For sentiment fine-tuning, set `"mode": "finetune"` and point `finetune.base_model` at the pretraining output.

### Benchmark

[`ModernFinTwitBERT/benchmark.py`](ModernFinTwitBERT/benchmark.py) scores the models in [`benchmark.json`](benchmark.json) on the test set:

- `"base"`: a pretrained model (Hugging Face id or local path) that is fine-tuned with the recipe in `config.json`, once per seed in `"seeds"`, then scored. Use this to compare starting points.
- `"classifier"`: an existing sentiment model that is scored as is. Add a `"label_map"` when its labels are not neutral/bullish/bearish or positive/negative, and a `"tokenizer"` when the model was uploaded without one.

`"finetune_args"` (top level or per model) overrides the fine-tuning arguments in `config.json`.

```bash
python ModernFinTwitBERT/benchmark.py
python ModernFinTwitBERT/benchmark.py --only fin-modernbert   # one entry
python ModernFinTwitBERT/benchmark.py --memory-check          # peak GPU memory on the longest tweets
```

Runs are appended to `output/benchmark/results.csv`, and runs already there are skipped, so adding a model and rerunning only runs the new one. `--rerun` runs the selected entries again. `output/benchmark/summary.md` has the mean and standard deviation per model. `macro_f1_low`/`macro_f1_high` is a 95% bootstrap interval: differences inside it are likely noise.

### Experiments

[`ModernFinTwitBERT/experiments.py`](ModernFinTwitBERT/experiments.py) fine-tunes the model with every setting in [`experiments.json`](experiments.json), once per seed, and logs each run to wandb (project `FinTwitRoBERTa-finetune`, one group per experiment).

```bash
python ModernFinTwitBERT/experiments.py
python ModernFinTwitBERT/experiments.py --only llrd-0.9 llrd-0.9-feed
```

- `"grid"` runs every combination of the listed TrainingArguments as its own experiment, e.g. `grid-lr2e-05-ep3`.
- `"llrd": 0.9` uses layer-wise learning-rate decay, and `"class_weights": "balanced"` weighs the loss by inverse class frequency. `"extra_train"` adds labeled files such as `feed_v1.csv` to the training split, with an optional `repeat`. Any other TrainingArguments (such as `label_smoothing_factor`) go in `"finetune_args"`.
- Pick the winner by dev macro F1 (`dev/macro_f1`) and use the test scores only to confirm it, so the test set is not tuned on. `val/f1` (a split of the training sources) is a poor guide: it ranked the first round of experiments almost in reverse order of test.
- `test/ece_calibrated` is the calibration error after temperature scaling on the dev set, and `ensemble macro_f1` scores the average of an experiment's seeds.
- Results go to `output/experiments/results.csv` and `summary.md`, and finished runs are skipped on a rerun.
- `--save` keeps a run's model with its dev-fitted temperature and a Hugging Face model card, and `--hub-repo` also uploads it. `python ModernFinTwitBERT/sentiment_card.py --experiment <name> --seed <seed>` rewrites the card of a trained model.

#### r/wallstreetbets experiments

[`experiments_wsb.json`](experiments_wsb.json) fine-tunes the WSB variant on [WSBS](https://github.com/moonscape95/WSBS) (downloaded on first use). It uses the same train/validation/test split of `WSB_all_agree.csv` as FinTwitBERT-wsb-sentiment, so both are tested on the same 227 posts. Experiments are picked by macro F1 on the 226 validation posts (`dev/macro_f1`), and `tweets/macro_f1` shows how well each still reads tweets. Results go to `output/experiments_wsb/`.

```bash
python ModernFinTwitBERT/experiments.py --config experiments_wsb.json
python ModernFinTwitBERT/wsb.py --score StephanAkkerman/FinTwitBERT-wsb-sentiment  # any model on the WSB test posts
```

- `from-sentiment` continues from FinTwitRoBERTa-sentiment (as the old WSB model did from FinTwitBERT-sentiment), and `from-large` starts from the pre-trained model.
- `from-sentiment-full` also trains on the 1,411 posts the annotators did not all agree on.
- `with-tweets` trains one model on the tweets plus the WSB posts (x5), to see if a single model can serve both.
- Posts are longer than tweets, so `max_length` is 256 (3% of posts are longer).

### Decision thresholds

[`ModernFinTwitBERT/thresholds.py`](ModernFinTwitBERT/thresholds.py) tunes how confident the model must be before a tweet counts as bullish or bearish. Per class, it picks the lowest threshold whose dev precision reaches `--target-precision` (default 0.9), confirms it on the test set, and writes the full tradeoff tables to `output/thresholds/`. Rerun it after every retrain.

```bash
python ModernFinTwitBERT/thresholds.py
python ModernFinTwitBERT/thresholds.py --model output/experiments/models/llrd-0.9-feed-seed42 --recompute --save
```

`--save` stores them in the model config as `decision_thresholds`, and `--hub-repo` uploads that config.

### ONNX export

[`ModernFinTwitBERT/export_onnx.py`](ModernFinTwitBERT/export_onnx.py) exports the model to ONNX with int8 weights (per-channel dynamic quantization). It checks on the dev and test sets that F1 is unchanged, and adds an "ONNX int8" section with size, speed and scores to the model card. `--hub-repo` uploads the result as `onnx/model_int8.onnx` with the updated card.

```bash
python ModernFinTwitBERT/export_onnx.py --hub-repo StephanAkkerman/FinTwitRoBERTa-sentiment
```

### Promoting the best model to Hugging Face

[`ModernFinTwitBERT/promote.py`](ModernFinTwitBERT/promote.py) runs the last steps in one go:

1. **Pick** the experiment with the best mean `dev/macro_f1` over its seeds, and its best seed (or pass `--experiment`/`--seed`). It warns when the lead is smaller than the seed spread.
2. **Build** the model: it reuses `output/.../models/<name>-seed<seed>` if that run was saved, and otherwise retrains it with `--save`.
3. **Check** it against the model on Hugging Face. The Hugging Face model is downloaded and both are scored on the same dev and test sets (for WSB, also on the tweet test set), so this works after local experiments are deleted and doesn't trust old numbers.
4. **Upload** only if the candidate's dev macro F1 beats the Hugging Face model's by more than `--min-gain` (default 0), or the repo doesn't exist yet. For the tweet model, the decision thresholds and the int8 ONNX model are then redone for the new weights and uploaded too.

The comparison is saved to `output/experiments*/promotion.json`.

```bash
python ModernFinTwitBERT/promote.py --dry-run                         # tweets, compare only
python ModernFinTwitBERT/promote.py                                   # FinTwitRoBERTa-sentiment
python ModernFinTwitBERT/promote.py --config experiments_wsb.json     # FinTwitRoBERTa-wsb-sentiment
```

## Citation

If you use FinTwitRoBERTa in your research, please cite it as follows:

```bibtex
@misc{FinTwitRoBERTa,
  author = {Stephan Akkerman},
  title = {FinTwitRoBERTa: Language Models for Financial Tweets},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/FinTwitRoBERTa}}
}
```

```bibtex
@misc{FinTwitRoBERTa-sentiment,
  author = {Stephan Akkerman},
  title = {FinTwitRoBERTa-sentiment: A Sentiment Classifier for Financial Tweets},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/StephanAkkerman/FinTwitRoBERTa-sentiment}}
}
```

FinTwitRoBERTa builds on [FinTwitBERT](https://github.com/TimKoornstra/FinTwitBERT) by Stephan Akkerman and Tim Koornstra, and the WSB model on the [WSBS](https://github.com/moonscape95/WSBS) dataset.

## Contributing

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.

![Contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/FinTwitRoBERTa)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
