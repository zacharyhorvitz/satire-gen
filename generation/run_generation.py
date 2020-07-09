#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

import re

# For analysis stuff
import json
import spacy
from pprint import pprint
import csv

nlp = spacy.load("en_core_web_lg")

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

def clean(headline):
    headline = headline.replace("....", ".").replace("|endoftext|", "").replace("<>", "").strip()
    if headline[:len("News:")] == "News:":
        headline=headline[len("News:"):]
    headline = headline.replace("'''", "").replace("[unused2]", "").replace("[unused1]", "").strip()

    return re.sub(r'[^\x00-\x7f]', r'', headline)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """"""

def add_to_dict(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]
    return True


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=1, top_p=0.0, is_xlnet=False, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet:
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def main():
    # Parsing args - don't touch these ##############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    ####### THIS IS THE SATIRICAL MODEL
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # model_satire = model_class.from_pretrained("../models/checkpoint-1200") #(args.model_name_or_path)
    # model_satire = model_class.from_pretrained("../models/checkpoint-1800")
    model_satire = model_class.from_pretrained("../models/checkpoint-1400")
    model_satire.to(args.device)
    model_satire.eval()
    ####### THIS IS THE NEWS MODEL
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_news = model_class.from_pretrained("../models/checkpoint-600") # (args.model_name_or_path)
    model_news.to(args.device)
    model_news.eval()
    ####### THIS IS THE NORMAL MODEL
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_og = model_class.from_pretrained("gpt2")
    # tokenizer = tokenizer_class.from_pretrained("gpt2-large") #args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained("gpt2") #args.model_name_or_path)
    model_og.eval()

    for i, model in enumerate([model_satire, model_news]):
        print(i, type(model))
        if args.length < 0 and model.config.max_position_embeddings > 0:
            args.length = model.config.max_position_embeddings
        elif 0 < model.config.max_position_embeddings < args.length:
            args.length = model.config.max_position_embeddings  # No generation bigger than model size
        elif args.length < 0:
            args.length = MAX_LENGTH  # avoid infinite loop

    print(args)

    ##################################################################################################################
    # THIS PART IS TO TAKE ALL THE ONION HEADLINES OUT THERE, TAKE THE FIRST TWO WORDS, AND THEN SEE THE DIFFERENCE  #
    # BETWEEN WHAT IS PRODUCED BY MODEL_NEWS, MODEL_SATIRE AND MODEL_OG                                              #
    ##################################################################################################################

    # At this point, we will open the tsv files
    # SATIRE_HEADLINE_PATH = "../data/all_test_heads.txt"
    # SATIRE_HEADLINE_PATH = "../data/all_news_heads_diff_types.txt"
    # with open(SATIRE_HEADLINE_PATH, "r") as satire_file:
    #     og_dict = {}
    #     for line in satire_file:
    #         converted = remove_non_ascii(line.strip())
    #         og_dict[converted] = True # dummy
    # SATIRE_HEADLINE_PATH = "../data/news_out_tsv.tsv"
    # with open(SATIRE_HEADLINE_PATH, "r", encoding="ISO-8859-1") as tsvfile:
    #     reader = csv.reader(tsvfile, delimiter="\t")
    #     og_dict = {}
    #     for row in reader:
    #         line = row[1]
    #         converted = remove_non_ascii(line.strip())
    #         if converted[:len("CNN 10")] == "CNN 10":
    #             continue
    #         og_dict[converted] = True # dummy
    og_dict = [
        "Biologists Confirm God Evolved From Chimpanzee Deity",
        "Coke-Sponsored Rover Finds Evidence Of Dasani On Mars",
        "Trump's Attacks On The Press",
        "Shrimp Boat Captain Worn Out From Long Day Of Putting Human Face On Crisis"
    ]

    dict_of_results = {}
    id = 0
    satire_dict, news_dict, gpt2_dict = {}, {}, {}
    # Loop through all the different satire headlines
    for satire_headline in og_dict:
        print("\n\n\n\n\n")
        id += 1
        doc = nlp(remove_non_ascii(satire_headline))
        first_two = " ".join([x.text for x in doc[:2]]).lower()
        if len(first_two.strip()) == 0: continue
        for time in range(1):
            if time > 2: break
            print(first_two)
            # and then generate per each
            this_iteration_dict = {} # {"model_satire": "Yeehaw is my fellows",
                                        # "model_news": "Yeehaw is the new hawyee"}
            # Deal with the two fine-tuned models, model_satire, model_news
            for i, model in enumerate([model_satire, model_news, model_og]):
                if i == 1: raw_text = "News: " + first_two.lower()
                else: raw_text = first_two.lower()
                # Unnecessito
                if args.model_type in ["transfo-xl", "xlnet"]:
                    # Models with memory likes to have a long prompt for short inputs.
                    raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
                # Encoding the context
                context_tokens = tokenizer.encode(raw_text)
                # Making the sequence
                out = sample_sequence(
                    model=model,
                    context=context_tokens,
                    length=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                    is_xlnet=bool(args.model_type == "xlnet"),
                )
                # out = out[0, len(context_tokens):].tolist()
                out = out[0, :].tolist()
                # Decoding the sequence
                text = clean(remove_non_ascii(tokenizer.decode(out, clean_up_tokenization_spaces=True)))
                if i == 0:
                    print("Satire: ", text)
                    add_to_dict(satire_dict, first_two, text)
                elif i == 1:
                    print("News: ", text)
                    add_to_dict(news_dict, first_two, text)
                else:
                    print("GPT2: ")
                    add_to_dict(gpt2_dict, first_two, text)
        # write every 50 heads we consider
        # if id%50 == 0:
    with open("../data/context_free_k1_p0_news_new_gpt2/for-zach-satire-gpt2-decoder.json", "w") as satire_file:
        json.dump(satire_dict, satire_file)
    with open("../data/context_free_k1_p0_news_new_gpt2/for-zach-news-gpt2-decoder.json", "w") as news_file:
        json.dump(news_dict, news_file)
    with open("../data/context_free_k1_p0_news_new_gpt2/for-zach-gpt2-gpt2-decoder.json", "w") as gpt2_file:
        json.dump(gpt2_dict, gpt2_file)

    print("Len satire_dict: ", len(satire_dict))
    print("Len news_dict: ", len(news_dict))

if __name__ == '__main__':
    main()
