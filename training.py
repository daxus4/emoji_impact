"""Fine tune a StockTwits sentiment classifier for one input condition."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from preprocessing import TransformSpec, load_sentiment_words, normalize_stocktwits_text, transform_text


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_TO_ID = {"bearish": 0, "bullish": 1}
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    condition: str
    action: str
    seed: int
    model_name: str = MODEL_NAME
    batch_size: int = 32
    max_length: int = 256
    max_epochs: int = 45
    unfreeze_epoch: int = 35
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    patience: int = 5
    early_stopping_min_delta: float = 0.001
    num_workers: int = 4


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("original", "emoji", "word", "emoji_word"))
    parser.add_argument("--action", choices=("replace", "remove"), default="replace")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=45)
    parser.add_argument("--unfreeze-epoch", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    settings = Settings(**vars(args))
    if settings.batch_size < 1 or settings.max_epochs < 1 or settings.patience < 1:
        parser.error("Batch size, epoch count, and patience must be positive")
    if not math.isfinite(settings.early_stopping_min_delta) or settings.early_stopping_min_delta < 0:
        parser.error("Early stopping min delta must be finite and nonnegative")
    return settings


class MessageDataset(Dataset):
    def __init__(self, path: Path, transform_spec: TransformSpec) -> None:
        self.examples: list[tuple[str, int, str]] = []
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError(f"Missing TSV header: {path}")
            text_column = "text" if "text" in reader.fieldnames else "Sentence"
            label_column = "Sentiment" if "Sentiment" in reader.fieldnames else "label"
            missing = {text_column, label_column} - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for row_number, row in enumerate(reader):
                label = LABEL_TO_ID.get(row[label_column].strip().casefold())
                if label is None:
                    raise ValueError(f"Invalid binary label in {path}: {row[label_column]!r}")
                text = row[text_column] if text_column == "text" else normalize_stocktwits_text(row[text_column])
                text = transform_text(text, transform_spec)
                example_id = row.get("source_row_id") or str(row_number)
                self.examples.append((text, label, example_id))
        if not self.examples:
            raise ValueError(f"Empty split: {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int, str]:
        return self.examples[index]


def make_collator(tokenizer: object, max_length: int):
    def collate(batch: list[tuple[str, int, str]]) -> dict[str, object]:
        texts, labels, example_ids = zip(*batch)
        encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {**encoded, "labels": torch.tensor(labels), "example_ids": example_ids}

    return collate


def make_loader(dataset: MessageDataset, settings: Settings, tokenizer: object, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        collate_fn=make_collator(tokenizer, settings.max_length),
    )


def focal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([2.0, 2.0], device=logits.device)
    cross_entropy = functional.cross_entropy(logits, labels, weight=weights, reduction="none")
    probability = torch.exp(-cross_entropy)
    return (((1 - probability) ** 2) * cross_entropy).mean()


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if key != "example_ids"}


def validation_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch_to_device(batch, device)
            labels = inputs.pop("labels")
            logits = model(**inputs).logits
            total_loss += focal_loss(logits, labels).item()
            batches += 1
    return total_loss / batches


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def make_grad_scaler(use_amp: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def train(model: torch.nn.Module, loaders: dict[str, DataLoader], settings: Settings, device: torch.device) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "classifier" in name or "pooler" in name
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("No classifier or pooler parameters found")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    total_steps = settings.max_epochs * len(loaders["train"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    use_amp = device.type == "cuda"
    scaler = make_grad_scaler(use_amp)
    best_loss = math.inf
    bad_epochs = 0
    forced_unfreeze = False
    unfreeze_next_epoch = False

    history_path = settings.output_dir / "history.jsonl"
    for epoch in range(settings.max_epochs):
        if epoch == settings.unfreeze_epoch or unfreeze_next_epoch:
            for parameter in model.parameters():
                parameter.requires_grad = True
            forced_unfreeze = forced_unfreeze or unfreeze_next_epoch
            unfreeze_next_epoch = False
        model.train()
        training_loss = 0.0
        halfway = math.ceil(len(loaders["train"]) / 2)
        for step, batch in enumerate(loaders["train"], start=1):
            inputs = batch_to_device(batch, device)
            labels = inputs.pop("labels")
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = focal_loss(model(**inputs).logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            training_loss += loss.item()
            if step == halfway:
                halfway_loss = validation_loss(model, loaders["validation"], device)
                print(json.dumps({"epoch": epoch, "step": step, "validation_loss": halfway_loss}), flush=True)
                model.train()

        current_loss = validation_loss(model, loaders["validation"], device)
        improved = best_loss - current_loss > settings.early_stopping_min_delta
        if improved:
            best_loss = current_loss
            bad_epochs = 0
            save_checkpoint(model, settings.output_dir / "best.pt")
        else:
            bad_epochs += 1
        save_checkpoint(model, settings.output_dir / "last.pt")
        record = {"epoch": epoch, "training_loss": training_loss / len(loaders["train"]),
                  "validation_loss": current_loss, "best_validation_loss": best_loss,
                  "bad_epochs": bad_epochs, "improved": improved}
        with history_path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if bad_epochs >= settings.patience:
            if epoch < settings.unfreeze_epoch and not forced_unfreeze and epoch + 1 < settings.max_epochs:
                unfreeze_next_epoch = True
                bad_epochs = 0
            else:
                break


def main() -> None:
    settings = parse_args()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    words = load_sentiment_words(Path(__file__).with_name("combined_sentiment_words.txt"))
    transform_spec = TransformSpec(settings.condition, settings.action, words)
    datasets = {name: MessageDataset(settings.data_dir / f"{name}.tsv", transform_spec) for name in SPLITS}
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    tokenizer.save_pretrained(settings.output_dir)
    loaders = {name: make_loader(dataset, settings, tokenizer, name == "train") for name, dataset in datasets.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    model.config.save_pretrained(settings.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (settings.output_dir / "settings.json").write_text(
        json.dumps({**asdict(settings), "data_dir": str(settings.data_dir), "output_dir": str(settings.output_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    train(model, loaders, settings, device)


if __name__ == "__main__":
    main()
