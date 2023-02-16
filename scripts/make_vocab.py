#!/usr/bin/env python3
# 2022 Cihan Xiao

import os
import argparse
from pathlib import Path
import logging
from collections import Counter
import re
import nltk.data

eng_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

extra_puncs = re.compile(
    r'\'|\\|\/|\*|\#|\$|\^|\&|\(|\)|\_|\+|\=|\"|\`|\~|\…|\.|\,|\:|\?|\!|\;|\-|\[|\]')
bracs = re.compile(r"|\<|\>")


def make_vocab(text, output, lang="en"):
    with open(text, "r") as f:
        lines = f.readlines()
        # Text will be normalized
        lines = [normalize_text(line, lang=lang) for line in lines if line.strip()]
        words = [line.strip().split() for line in lines]
        words = [word for line in words for word in line]
        word_counts = Counter(words)
        word_counts = sorted(word_counts.items(), key=lambda x: x[0])
        with open(output, "w") as f:
            for i, (word, count) in enumerate(word_counts):
                print(i + 1, word, count, file=f)


def normalize_text(text, lang="en"):
    if lang == "en":
        text = eng_tokenizer.tokenize(text)
    text = [t.lower() for t in text] if lang == "en" else [t.lower() for t in text.split()]
    # Convert between-digit puncs to special tokens to prevent removing them
    text = [re.sub(r"(\d)\.(\d)", r"\1\<dot\>\2", t.strip()) for t in text]
    text = [re.sub(r"(\d)\,(\d)", r"\1\<com\>\2", t.strip()) for t in text]
    text = [re.sub(r"(\d)\:(\d)", r"\1\<col\>\2", t.strip()) for t in text]

    text = [re.sub(extra_puncs, "", t.strip()) for t in text]

    text = [re.sub("<dot>", ".", t.strip()) for t in text]
    text = [re.sub("<com>", ",", t.strip()) for t in text]
    text = [re.sub("<dot>", ":", t.strip()) for t in text]

    text = [re.sub(bracs, "", t.strip()) for t in text]

    text = [re.sub("-", " - ", t.strip()) for t in text]
    text = [re.sub("%", " % ", t.strip()) for t in text]

    # Remove the extra spaces
    text = [re.sub(r"\s+", " ", t.strip()) for t in text]
    return " ".join(text) if len(text) >= 1 else ""


def _normalize_text(line, lang="en"):
    # Add space before ...
    line = re.sub(r"\.\.\.", r" ...", line)
    line = re.sub(r"([\(\)\?\:\,\"\'\!])", r" \1 ", line)
    # Remove extra spaces
    line = re.sub(r"\s+", " ", line)
    # Add space before period if nothing follows it
    line = re.sub(r"\.([\s|$|\"])", r" .\1", line)
    # Add space before % if it has digit before it
    line = re.sub(r"(\d)\%", r"\1 %", line)
    # Merge .. .
    line = re.sub(r"\.\s*\.\s*\.", r" ...", line)
    # Merge ' s for English
    if lang == "en":
        line = re.sub(r"\' s\s+", r" 's ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Make a vocabulary file from a text file.')
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='The full path to the source text file.')
    parser.add_argument('--output', type=Path, required=True,
                        help='The output vocabulary file.')
    parser.add_argument('--lang', default="en", choices=["en", "sv"],
                        help='The language of the source text file.')
    args = parser.parse_args()
    make_vocab(text=args.input, output=args.output, lang=args.lang)


if __name__ == "__main__":
    main()
