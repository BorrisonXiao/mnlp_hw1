#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Cihan Xiao 2022

import argparse
from pathlib import Path
import re
import pyconll
import conllu
from conllu import TokenList, SentenceList, Token

puncs = re.compile(
    r'\'|\\|\/|\*|\#|\$|\^|\&|\(|\)|\_|\+|\=|\"|\`|\~|\…|\.|\,|\:|\?|\!|\;|\-|\[|\]')


def parse_alignment(line, verbose=False):
    """
    Helper function to parse the alignment line.
    :param line: The alignment line
    :param verbose: Whether to print the alignment
    :return: A list of alignments, each alignment is a dict with three keys: token, src_idx and tgt_indices
    """
    res = []
    alignment = line.strip()
    # Parse the alignment whose format is <token> ({ <indx> })
    # For example, "This ({ 1 })"
    # The first token is the source token, the second index is the index of the target token
    # The index starts from 1
    # The alignment is 1-to-many
    alignments = alignment.split()
    src_idx = 0
    while len(alignments) > 0:
        # Skip the NULL alignments
        if alignments[0] == "NULL":
            while alignments[0] != "})":
                alignments.pop(0)
            alignments.pop(0)
            continue

        src_token = alignments.pop(0)
        # Skip the "({" token
        alignments.pop(0)
        # Get indices of the target tokens
        tgt_indices = []
        while alignments[0] != "})":
            tgt_indices.append(int(alignments.pop(0)) - 1)

        # Skip the "})" token
        alignments.pop(0)
        # Add the alignment to the result
        res.append(dict(token=src_token, src_idx=src_idx,
                   tgt_indices=tgt_indices))
        if verbose:
            print(f"{src_token} -> {tgt_indices}")
        src_idx += 1
    return res


def reverse_alignment(alignments, tgt_tokens, verbose=False):
    """
    Reverse the alignments.
    :param alignments: The alignments to be reversed
    :return: A dict of reversed alignments, i.e. the key is the target index (and token) and the value is the source index (and token)
    """
    res = {}
    for alignment in alignments:
        src_idx = alignment['src_idx']
        src_token = alignment['token']
        for tgt_idx in alignment['tgt_indices']:
            tgt_token = tgt_tokens[tgt_idx]
            assert tgt_idx not in res
            res[tgt_idx] = dict(tgt_token=tgt_token, src_idx=src_idx, src_token=src_token)
    # Sort the alignments by the target index
    res = dict(sorted(res.items(), key=lambda item: item[0]))
    if verbose:
        print("Reversed alignments:")
        for tgt_idx, alignment in res.items():
            print(f"{tgt_idx} {alignment['tgt_token']} -> {alignment['src_token']}")
        print()

    return res


def write_conllu(output: Path, sentences: SentenceList):
    """
    Write the sentences to a CoNLL-U file.
    :param output: The output file
    :param sentences: The sentences to be written
    """
    with open(output, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                f.write(sentence.serialize())
            except:
                breakpoint()


def project(src: Path, A1_file: Path, start: int, output: Path, verbose: bool = False):
    alignments = []
    # First parse the A1 alignment file
    with open(A1_file, 'r', encoding='utf-8') as f:
        # Read every three lines
        for i, line in enumerate(f):
            # if i == 5:
            #     assert False

            score = float(line.strip().split(" : ")[-1])
            line = next(f)
            tgt_text = line.strip()
            # # Punctuation will be break into a separate token
            # tgt_text = re.sub(puncs, " \\g<0>", tgt_text)
            tgt_tokens = tgt_text.split()
            line = next(f)
            alignment = parse_alignment(line)
            if verbose:
                print(f"Score: {score}")
                print(f"Target: {tgt_text}")
                print(f"Alignments:")
                for align in alignment:
                    print(
                        f"{align['src_idx']} {align['token']} ( {[tgt_tokens[idx] for idx in align['tgt_indices']]} )")
                print()
            rev_alignment = reverse_alignment(alignment, tgt_tokens)
            rev_alignment['tgt_text'] = tgt_text
            alignments.append(rev_alignment)
    
    alignments = alignments[start:]

    parsed = pyconll.load_from_file(src)
    sentences = SentenceList()
    for src_sentence, alignment in zip(parsed, alignments):
        # Create a new sentence based on the alignment, preserving the attributes except for the token form and lemma
        tgt_sentence = TokenList()
        tgt_sentence.metadata = src_sentence._meta
        tgt_sentence.metadata['text'] = alignment['tgt_text']
        for tgt_idx, tgt_tok in enumerate(alignment['tgt_text'].split()):
            if tgt_idx not in alignment:
                # This is a new token
                tgt_token = Token()
                tgt_token['id'] = tgt_idx + 1
                tgt_token['form'] = tgt_tok
                tgt_token['lemma'] = "_"
                tgt_token['upos'] = "_"
                tgt_token['xpos'] = "_"
                tgt_token['head'] = "_"
                tgt_token['deprel'] = "_"
                tgt_token['deps'] = "_"
                tgt_token['misc'] = "_"
                tgt_sentence.append(tgt_token)
                continue
            # TODO: This has a bug, punctuations are not breaking into separate tokens
            src_idx = alignment[tgt_idx]['src_idx']
            src_token = src_sentence[src_idx]
            tgt_token = Token()
            tgt_token['id'] = tgt_idx + 1
            tgt_token['form'] = tgt_tok
            tgt_token['lemma'] = "_"
            tgt_token['upos'] = src_token.upos
            tgt_token['xpos'] = src_token.xpos
            # feats are normalized to strings
            norm_feats = "|".join([f"{k}={next(iter(v))}" for k, v in src_token.feats.items()]) if isinstance(src_token.feats, dict) else src_token.feats
            tgt_token['feats'] = norm_feats
            # tgt_token['head'] = src_token.head
            # tgt_token['deprel'] = src_token.deprel
            # tgt_token['deps'] = src_token.deps
            # # misc will be normalized as well
            # norm_misc = "|".join([f"{k}={next(iter(v))}" for k, v in src_token.misc.items()]) if isinstance(src_token.misc, dict) else src_token.misc
            # tgt_token['misc'] = norm_misc
            tgt_token['head'] = "_"
            tgt_token['deprel'] = "_"
            tgt_token['deps'] = "_"
            tgt_token['misc'] = "_"
            tgt_sentence.append(tgt_token)
        sentences.append(tgt_sentence)

    write_conllu(output, sentences)


def main():
    """
    Project the conllu file based on the GIZA++ alignment output.
    """

    parser = argparse.ArgumentParser(
        description='Project the conllu file based on the GIZA++ alignment output.')
    parser.add_argument('--src', type=Path, required=True,
                        help='The full path to the source conllu file.')
    parser.add_argument('--start', type=int, default=0,
                        help='The start index of the alignment file.')
    parser.add_argument('--A1-file', type=Path, required=True,
                        help='The full path to the GIZA++ alignment output file.')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the output file.')
    args = parser.parse_args()
    project(src=args.src, A1_file=args.A1_file, start=args.start, output=args.output)


if __name__ == "__main__":
    main()
