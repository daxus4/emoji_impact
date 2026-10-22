# Emoji and sentiment word ablation experiments

This repository fine tunes a binary RoBERTa sentiment classifier under four input conditions: `original`, `emoji`, `word`, and `emoji_word`. For the ablated conditions, choose `replace` or `remove`. Dataset splitting is outside this repository.

## Input data

Provide one directory with tab separated training, validation, and test files:

```
data/
  train.tsv
  validation.tsv
  test.tsv
```

Each file needs a `Sentiment` column containing `Bullish` or `Bearish`, plus either `text` (already normalized StockTwits text) or `Sentence` (raw text to normalize). A `source_row_id` column is optional. When it is absent, the prediction file uses the zero based row number within the supplied test file as `example_id`.

Pass the directory with `--data-dir` to `training.py`; it reads `train.tsv` and `validation.tsv`. Pass any labeled test TSV to `testing.py` with `--test-file`. The programs do not create splits or check their IDs against [`split_ids.csv.gz`](split_ids.csv.gz). The archive remains available as a separate reference, without being required to run the scripts.

`combined_sentiment_words.txt` contains the word list used for the word ablations and must remain alongside the Python files.

## Run

From this directory, create and activate the Conda environment on Linux or Windows:

```bash
conda env create -f environment.yml
conda activate paper-finetuning
python training.py --data-dir data --output-dir runs/example_run --condition emoji --action replace --seed 0
python testing.py --test-file data/test.tsv --model-dir runs/example_run --output-dir runs/example_test_emoji_replace --condition emoji --action replace
```

The environment file pins Python 3.13.2, the original project's runtime dependencies, and the NumPy and Hugging Face package versions resolved by `uv run`. It has no machine-specific `prefix`. On systems without Conda, install the same pinned Python packages with `pip install -r requirements.txt` in a Python 3.13.2 environment.

`--seed` controls model initialization, training order, and other random operations. It does not select a dataset. To use another split, point `--data-dir` to another directory containing `train.tsv` and `validation.tsv`. Testing loads the checkpoint with the lowest validation loss from `best.pt`; it never retrains the model. To use an all-emoji test set, pass its TSV path with `--test-file`.

Each `testing.py` run evaluates exactly one input variant. Choose `--condition original` for unchanged text, or choose `emoji`, `word`, or `emoji_word` together with `--action remove` or `--action replace`. Run the command again with a different output directory for another variant. It writes `metrics.json`, `metrics.csv`, and `predictions.csv` to the chosen output directory. The reports include MCC, accuracy, macro F1, negative F1, and positive F1. In the prediction CSV, `0` means Bearish/negative and `1` means Bullish/positive; each row includes the scenario, example ID, predicted class, and both class probabilities.

## Processing and training

Raw `Sentence` input has URLs removed, HTML apostrophe entities decoded, hashtags/cashtags/mentions rewritten, and whitespace collapsed. Normalized `text` input skips that first step. Emoji ablation replaces each emoji with `[EMOJI]` or removes it; word ablation matches case insensitively against `combined_sentiment_words.txt` and replaces with `[WORD]` or removes the matched word. Inputs are tokenized with `cardiffnlp/twitter-roberta-base-sentiment`, maximum length 256.

The classifier starts with a new two class head. Training uses focal loss (alpha `[2, 2]`, gamma 2), AdamW (learning rate `5e-5`, weight decay `0.01`), batch size 32, a linear schedule without warmup, gradient clipping at 1, half epoch and end of epoch validation, and up to 45 epochs. Initially only the classifier and pooler train. All layers unfreeze at epoch index 35, or earlier when patience 5 is exhausted. Early stopping uses a minimum validation loss improvement of `0.001` by default, configurable with `--early-stopping-min-delta`. The best checkpoint is selected by validation loss under that threshold. GPU runs use mixed precision; CPU runs use full precision.

Training output contains `settings.json`, `history.jsonl`, `best.pt`, `last.pt`, and the saved model configuration and tokenizer. Testing output contains `metrics.json`, `metrics.csv`, and `predictions.csv`.
