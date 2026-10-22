"""Evaluate one fine tuned checkpoint on the seven StockTwits input conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import TransformSpec, load_sentiment_words
from training import MessageDataset, batch_to_device, make_collator


CONDITIONS = ("emoji", "word", "emoji_word")
ACTIONS = ("remove", "replace")
PREDICTION_COLUMNS = (
    "condition", "action", "example_id", "label", "prediction",
    "negative_probability", "positive_probability",
)
METRIC_COLUMNS = ("condition", "action", "n_samples", "mcc", "accuracy", "macro_f1", "negative_f1", "positive_f1")


@dataclass(frozen=True)
class Settings:
    test_file: Path
    model_dir: Path
    output_dir: Path
    condition: str = "all"
    action: str = "both"
    batch_size: int = 32
    num_workers: int = 4


@dataclass(frozen=True)
class Scenario:
    condition: str
    action: str | None = None

    @property
    def name(self) -> str:
        return self.condition if self.action is None else f"{self.condition}_{self.action}"


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", choices=("all", "original", *CONDITIONS), default="all")
    parser.add_argument("--action", choices=("both", *ACTIONS), default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    settings = Settings(**vars(parser.parse_args()))
    if settings.batch_size < 1 or settings.num_workers < 0:
        parser.error("Batch size must be positive and worker count nonnegative")
    if not settings.test_file.is_file():
        parser.error(f"Test file does not exist: {settings.test_file}")
    for filename in ("settings.json", "config.json", "best.pt"):
        if not (settings.model_dir / filename).is_file():
            parser.error(f"Trained model is missing {filename}: {settings.model_dir}")
    return settings


def selected_scenarios(settings: Settings) -> tuple[Scenario, ...]:
    if settings.condition == "original":
        return (Scenario("original"),)
    conditions = CONDITIONS if settings.condition == "all" else (settings.condition,)
    actions = ACTIONS if settings.action == "both" else (settings.action,)
    ablations = tuple(Scenario(condition, action) for condition in conditions for action in actions)
    return (Scenario("original"), *ablations) if settings.condition == "all" else ablations


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)
    state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def summarize_predictions(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    negative_denominator = 2 * true_negative + false_positive + false_negative
    positive_denominator = 2 * true_positive + false_positive + false_negative
    negative_f1 = 2 * true_negative / negative_denominator if negative_denominator else 0.0
    positive_f1 = 2 * true_positive / positive_denominator if positive_denominator else 0.0
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (true_positive * true_negative - false_positive * false_negative) / mcc_denominator if mcc_denominator else 0.0
    return {
        "n_samples": len(labels),
        "mcc": mcc,
        "accuracy": (true_positive + true_negative) / len(labels),
        "macro_f1": (negative_f1 + positive_f1) / 2,
        "negative_f1": negative_f1,
        "positive_f1": positive_f1,
    }


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device, writer: csv.writer) -> None:
        self.model = model
        self.device = device
        self.writer = writer

    def evaluate(self, scenario: Scenario, loader: DataLoader) -> dict[str, float | int | str]:
        labels: list[int] = []
        predictions: list[int] = []
        with torch.inference_mode():
            for batch in loader:
                inputs = batch_to_device(batch, self.device)
                expected = inputs.pop("labels").cpu().tolist()
                probabilities = self.model(**inputs).logits.softmax(dim=-1).cpu()
                predicted = probabilities.argmax(dim=-1).tolist()
                labels.extend(expected)
                predictions.extend(predicted)
                for example_id, label, prediction, probability in zip(
                    batch["example_ids"], expected, predicted, probabilities.tolist()
                ):
                    self.writer.writerow((
                        scenario.condition, scenario.action or "none", example_id,
                        label, prediction, *probability,
                    ))
        return {
            "condition": scenario.condition,
            "action": scenario.action or "none",
            **summarize_predictions(labels, predictions),
        }


def write_reports(output_dir: Path, results: dict[str, dict[str, float | int | str]]) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(results.values())


def main() -> None:
    settings = parse_args()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    training_settings = json.loads((settings.model_dir / "settings.json").read_text(encoding="utf-8"))
    max_length = int(training_settings["max_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True)
    model = load_model(settings.model_dir, device)
    words = load_sentiment_words(Path(__file__).with_name("combined_sentiment_words.txt"))
    results: dict[str, dict[str, float | int | str]] = {}
    with (settings.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(PREDICTION_COLUMNS)
        evaluator = Evaluator(model, device, writer)
        for scenario in selected_scenarios(settings):
            spec = TransformSpec(scenario.condition, scenario.action or "replace", words)
            dataset = MessageDataset(settings.test_file, spec)
            loader = DataLoader(
                dataset,
                batch_size=settings.batch_size,
                num_workers=settings.num_workers,
                collate_fn=make_collator(tokenizer, max_length),
            )
            results[scenario.name] = evaluator.evaluate(scenario, loader)
            print(json.dumps(results[scenario.name]), flush=True)
    write_reports(settings.output_dir, results)


if __name__ == "__main__":
    main()
