#!/home/cxiao7/miniconda3/envs/mnlp/bin/python
# -*- coding: utf-8 -*-
# Cihan Xiao 2022

import argparse
from pathlib import Path
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

def translate(modelname: str, src: Path, output: Path):
    # # A hack for rerunning only certain text
    # rerun_list = [11, 16, 34, 41, 57, 69]
    # # Get the src index
    # src_idx = int(str(src).split('.')[-2])
    # if src_idx not in rerun_list:
    #     print(f"{src} is not in the rerun list. Skip.")
    #     return

    # First load the model and the tokenizer
    model = M2M100ForConditionalGeneration.from_pretrained(modelname)
    tokenizer = M2M100Tokenizer.from_pretrained(modelname)

    
    tokenizer.src_lang = "en"
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output, 'w', encoding='utf-8') as f:
        for line in tqdm(lines):
            encoded = tokenizer(line, return_tensors="pt")
            generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("sv"))
            hyp = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(hyp[0], file=f)


def main():
    """
    Translate the English text into Swedish using m2m-100.
    """

    parser = argparse.ArgumentParser(
        description='Translate the English text into Swedish using m2m-100.')
    parser.add_argument('--model', type=str, default="facebook/m2m100_418M",
                        help='The full of the m2m model to use.')
    parser.add_argument('--src', type=Path, required=True,
                        help='The full path to the file to be translated.')
    parser.add_argument('--output', type=Path, required=True,
                        help='The full path to the output file.')
    args = parser.parse_args()
    translate(modelname=args.model, src=args.src, output=args.output)


if __name__ == "__main__":
    main()
