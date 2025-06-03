import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime
import json

import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    from utils import add_common_args
    add_common_args(p)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=64*1024, help="The length of the context.")
    p.add_argument("--result_dir", type=str, default="results")
    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name_or_path

    seed_everything(42)
    
    from utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_flash_attn2=args.flash2)
    
    
    if args.xKV:
        from utils import apply_kv_compress_patch
        model_patch = apply_kv_compress_patch(model, args)

    prompt = ["He smiled understandingly—much more than understandingly. It was one of those rare smiles with a quality of eternal reassurance in it, that you may come across four or five times in life. It faced—or seemed to face—the whole external world for an instant, and then concentrated on YOU with an irresistible prejudice in your favor. It understood you just so far as you wanted to be understood, believed in you as you would like to believe in yourself and assured you that it had precisely the impression of you that, at your best, you hoped to convey. Precisely at that point it vanished— and I was looking at an elegant young rough-neck, a year or two over thirty, whose elaborate formality of speech just missed being absurd. Some time before he introduced himself I’d got a strong impression that he was picking his words with care."]
    input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_patch.generate(**(input.to("cuda")), max_new_tokens=1)

    from IPython import embed
    # embed()
    
    
    del model
