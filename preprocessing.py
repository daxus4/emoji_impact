"""StockTwits text preparation used in the paper's four input conditions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import emoji


URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
APOSTROPHE_PATTERN = re.compile(r"(?:&#0*39;|&#x0*27;|&apos;)", re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r"(?<!\w)#(\w+)", re.UNICODE)
CASHTAG_PATTERN = re.compile(r"(?<!\w)\$([A-Za-z][A-Za-z0-9._-]*)")
MENTION_PATTERN = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TransformSpec:
    condition: str
    action: str
    sentiment_words: frozenset[str]


def load_sentiment_words(path: Path) -> frozenset[str]:
    words = frozenset(
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not words:
        raise ValueError(f"Empty sentiment word list: {path}")
    return words


def normalize_stocktwits_text(text: str) -> str:
    value = URL_PATTERN.sub(" ", text)
    value = APOSTROPHE_PATTERN.sub("'", value)
    value = HASHTAG_PATTERN.sub(r"hashtag_\1", value)
    value = CASHTAG_PATTERN.sub(r"cashtag_\1", value)
    value = MENTION_PATTERN.sub(r"mention_\1", value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def replace_sentiment_words(text: str, words: frozenset[str], replacement: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        return replacement if match.group(0).casefold() in words else match.group(0)

    return WORD_PATTERN.sub(replace_match, text)


def transform_text(normalized_text: str, spec: TransformSpec) -> str:
    if spec.condition not in {"original", "emoji", "word", "emoji_word"}:
        raise ValueError(f"Unknown condition: {spec.condition}")
    if spec.action not in {"replace", "remove"}:
        raise ValueError(f"Unknown ablation action: {spec.action}")

    text = normalized_text
    if spec.condition in {"emoji", "emoji_word"}:
        token = "[EMOJI]" if spec.action == "replace" else ""
        text = emoji.replace_emoji(text, replace=f" {token} " if token else " ")
        text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if spec.condition in {"word", "emoji_word"}:
        token = "[WORD]" if spec.action == "replace" else ""
        text = replace_sentiment_words(text, spec.sentiment_words, token)
        if not token:
            text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
