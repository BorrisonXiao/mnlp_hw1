#!/usr/bin/env python3
# 2022 Cihan Xiao

import argparse
from pathlib import Path
import logging
from collections import Counter
from tqdm import tqdm
from make_vocab import normalize_text


def parse_vocab(vocab):
    token2id = {}
    with open(vocab, "r") as f:
        lines = f.readlines()
    for line in lines:
        id, token, _ = line.strip().split()
        token2id[token] = id
    return token2id


def make_bitext(src, tgt, src_lang, tgt_lang, src_vocab, tgt_vocab, output):
    counter = Counter()
    src_token2id = parse_vocab(src_vocab)
    tgt_token2id = parse_vocab(tgt_vocab)
    with open(src, "r") as f:
        src_lines = f.readlines()
        src_lines = [normalize_text(line, lang=src_lang)
                     for line in src_lines]
        # Convert the tokens to ids
        src_lines = [" ".join([src_token2id[token] for token in line.split()]) for line in src_lines]
    # Count the occurrences of each sentence
    for line in src_lines:
        counter[line] += 1
    with open(tgt, "r") as f:
        tgt_lines = f.readlines()
        tgt_lines = [normalize_text(line, lang=tgt_lang)
                     for line in tgt_lines]
        # Convert the tokens to ids
        tgt_lines = [" ".join([tgt_token2id[token] for token in line.split()]) for line in tgt_lines]
    assert len(src_lines) == len(tgt_lines)
    with open(output, "w") as f:
        for src_line, tgt_line in tqdm(zip(src_lines, tgt_lines)):
            if src_line == "" or tgt_line == "":
                continue
            occurrences = counter[src_line] if len(src_line) > 0 else 1
            print(occurrences, file=f)
            print(src_line, file=f)
            print(tgt_line, file=f)


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Generate the bitext file for GIZA++.')
    parser.add_argument('--src', type=Path, required=True,
                        help='The full path to the source text file.')
    parser.add_argument('--tgt', type=Path, required=True,
                        help='The full path to the target text file.')
    parser.add_argument('--src-lang', default="en", choices=["en", "sv"],
                        help='The language of the source text file.')
    parser.add_argument('--tgt-lang', default="sv", choices=["en", "sv"],
                        help='The language of the target text file.')
    parser.add_argument('--src-vocab', type=Path, required=True,
                        help='The vocab of the source text file.')
    parser.add_argument('--tgt-vocab',type=Path, required=True,
                        help='The vocab of the target text file.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='The full path to the output bitext file.')
    args = parser.parse_args()
    make_bitext(src=args.src, tgt=args.tgt, src_lang=args.src_lang,
                src_vocab=args.src_vocab, tgt_vocab=args.tgt_vocab,
                tgt_lang=args.tgt_lang, output=args.output)


if __name__ == "__main__":
    main()
