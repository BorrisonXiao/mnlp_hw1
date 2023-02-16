#!/usr/bin/env python3
# 2022 Cihan Xiao

import argparse
from pathlib import Path
import logging
from make_vocab import _normalize_text
import pyconll


def make_bitext(src, conllu_file, limit, output):
    src_lang = Path(src).suffix[1:]
    if conllu_file is not None:
        parsed = pyconll.load_from_file(conllu_file)
        # The src_lines will be the tokenized sentences
        src_lines = []
        for i, sentence in enumerate(parsed):
            src_line = " ".join([t.form for t in sentence])
            src_lines.append(src_line)
        # The rest will be automatically tokenized
        with open(src, "r") as f:
            for j, line in enumerate(f):
                if j <= i:
                    continue
                src_line = _normalize_text(line, lang=src_lang)
                src_lines.append(src_line)
    else:
        with open(src, "r") as f:
            src_lines = f.readlines()
            if limit < 0:
                limit = len(src_lines)
            src_lines = [_normalize_text(line, lang=src_lang)
                        for line in src_lines[:limit]]
    with open(output, "w") as f:
        for src_line in src_lines:
            print(src_line, file=f)


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Generate the bitext file for GIZA++.')
    parser.add_argument('--input', type=Path, required=True,
                        help='The full path to the source text file.')
    parser.add_argument('--conllu-file', type=Path, default=None,
                        help='The full path to the conllu file. If provided, the tokenizatio will be based on this file.')
    parser.add_argument('--limit', type=int, default=-1,
                        help='The number of lines to read from the input file.')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='The full path to the output bitext file.')
    args = parser.parse_args()
    make_bitext(src=args.input, conllu_file=args.conllu_file, limit=args.limit, output=args.output)


if __name__ == "__main__":
    main()
