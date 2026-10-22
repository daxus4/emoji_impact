# Emoji and sentiment word ablation experiments

This repository fine tunes a binary RoBERTa sentiment classifier under four input conditions: `original`, `emoji`, `word`, and `emoji_word`. For the ablated conditions, choose `replace` or `remove`. Dataset splitting is outside this repository.

## Input data

Provide one directory with three tab separated files:

```
data/
  train.tsv
  validation.tsv
  test.tsv
```

Each file needs a `Sentiment` column containing `Bullish` or `Bearish`, plus either `text` (already normalized StockTwits text) or `Sentence` (raw text to normalize). A `source_row_id` column is optional. When it is absent, prediction files use the zero based row number within `test.tsv` as `example_id`.

Pass the directory with `--data-dir`. The program reads the three files as supplied: it does not create splits or check their IDs against [`split_ids.csv.gz`](split_ids.csv.gz). The archive remains available as a separate reference, without being required for training.

`combined_sentiment_words.txt` contains the word list used for the word ablations and must remain alongside the Python files.

## Run

From this directory, create and activate the Conda environment on Linux or Windows:

```bash
conda env create -f environment.yml
conda activate paper-finetuning
python experiment.py --data-dir data --output-dir runs/example_run --condition emoji --action replace --seed 0
```

The environment file has no machine-specific `prefix`. On systems without Conda, install the same pinned Python packages with `pip install -r requirements.txt` in a Python 3.11 environment.

`--seed` controls model initialization, training order, and other random operations. It does not select a dataset. To use another split, point `--data-dir` to another directory containing the same three filenames. The best validation checkpoint is evaluated on `test.tsv`.

## Processing and training

Raw `Sentence` input has URLs removed, HTML apostrophe entities decoded, hashtags/cashtags/mentions rewritten, and whitespace collapsed. Normalized `text` input skips that first step. Emoji ablation replaces each emoji with `[EMOJI]` or removes it; word ablation matches case insensitively against `combined_sentiment_words.txt` and replaces with `[WORD]` or removes the matched word. Inputs are tokenized with `cardiffnlp/twitter-roberta-base-sentiment`, maximum length 256.

The classifier starts with a new two class head. Training uses focal loss (alpha `[2, 2]`, gamma 2), AdamW (learning rate `5e-5`, weight decay `0.01`), batch size 32, a linear schedule without warmup, gradient clipping at 1, half epoch and end of epoch validation, and up to 45 epochs. Initially only the classifier and pooler train. All layers unfreeze at epoch index 35, or earlier when patience 5 is exhausted. The best checkpoint is selected by validation loss. GPU runs use mixed precision; CPU runs use full precision.

Each output directory contains `settings.json`, `history.jsonl`, `best.pt`, `last.pt`, `metrics.json`, and `test_predictions.csv`.
