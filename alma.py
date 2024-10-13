import time

import torch
import torch.nn as nn
import os

from gptq import *
from modelutils import *
from quant import *
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, GPTQConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from optimum.gptq import GPTQQuantizer, load_quantized_model

from datasets import Dataset, concatenate_datasets
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from typing import List



def get_alma(nsamples: int, seed: int, seqlen: int, source_lang: str, target_lang: str) -> List[str]:
    # Define all language directions
    full_alma_splits = {
        'train': {
            'cs-en': 'cs-en/train-00000-of-00001-3a60b130a713425b.parquet',
            'de-en': 'de-en/train-00000-of-00001-39460826cd7ac756.parquet',
            'ru-en': 'ru-en/train-00000-of-00001-3ba3fad04eea46f0.parquet',
            'zh-en': 'zh-en/train-00000-of-00001-6bd744feceb30dbf.parquet'
        }
    }

    LANG_MAP = {
        'cs': 'Czech', 'de': 'German', 'en': 'English', 'ru': 'Russian', 'zh': 'Chinese'
    }
   
    # Set seed for reproducibility
    random.seed(seed)

    all_prompts = []

    # Iterate over all language directions
    for lang_pair, file_path in full_alma_splits['train'].items():
        print(f"Loading dataset for {lang_pair}")

        # Load train split
        train_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{file_path}"
        print("Train link: ", train_split_path)
        train_df = pd.read_parquet(train_split_path)

        # Convert DataFrame to Hugging Face Dataset
        traindata = Dataset.from_pandas(train_df)

        src_lang, tgt_lang = lang_pair.split('-')

        # Sample from the dataset
        sampled_data = traindata.shuffle(seed=seed).select(range(min(nsamples, len(traindata))))

        # Create prompts
        for example in sampled_data:
            source_text = example['translation'][src_lang]
            prompt = (
                f"Translate this from {LANG_MAP[src_lang]} to {LANG_MAP[tgt_lang]}:\n"
                f"{LANG_MAP[src_lang]}: {source_text}\n"
                f"{LANG_MAP[tgt_lang]}:"
            )
            
            # Only add prompts that don't exceed the seqlen
            if len(prompt.split()) <= seqlen:
                all_prompts.append(prompt)
            # print('all_prompts', all_prompts)

    # Shuffle the combined prompts
    random.shuffle(all_prompts)
    
    # Limit to nsamples if we have more
    return all_prompts[:nsamples]

def get_loaders_alma(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, source_lang="cs", target_lang="en"):
    if "parallel" in name:
        return get_alma(nsamples, seed, seqlen, source_lang, target_lang)
        
    

def quantize_alma(model):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    # from transformers import LlamaForCausalLM

    if args.local_path:
        ##get the tokenizer from the corresponding hf directory
        tokenizer = LlamaTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left')
        dataloader = get_loaders_alma(
            "parallel",
            nsamples=512, 
            seed=0, 
            seqlen=2048, 
            tokenizer=tokenizer,
            source_lang='en',
            target_lang='de'
        )
        print("dataset loading complete")

        gptq_config = GPTQConfig(bits=args.wbits, dataset=dataloader, tokenizer=tokenizer, use_cuda_fp16=True)
        ##quantize the model using the local_path argument
        quantized_model = AutoModelForCausalLM.from_pretrained(args.local_path, device_map="auto", quantization_config=gptq_config)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left')
        dataloader = get_loaders_alma(
            "parallel",
            nsamples=512, 
            seed=0, 
            seqlen=2048, 
            tokenizer=tokenizer,
            source_lang='en',
            target_lang='de'
        )
        print("dataset loading complete")

        gptq_config = GPTQConfig(bits=args.wbits, dataset=dataloader, tokenizer=tokenizer)
        quantized_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=gptq_config)

    return quantized_model



# def quantize_alma_autgptq(model):

#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip
#     # from transformers import LlamaForCausalLM
#     tokenizer = AutoTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left')

#     examples = [
#         tokenizer(
#             "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
#         )
#     ]


#     quantize_config = BaseQuantizeConfig(
#         bits=args.wbits,  # quantize model to 4-bit
#         group_size=128,  # it is recommended to set the value to 128
#         desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
#     )

#     # load un-quantized model, by default, the model will always be loaded into CPU memory
#     model = AutoGPTQForCausalLM.from_pretrained(args.model, quantize_config, device_map="auto")

#     # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
#     model.quantize()

    
#     return model
# def quantize_gptqquantizer(model):

#     def skip(*args, **kwargs):
#         pass
#     torch.nn.init.kaiming_uniform_ = skip
#     torch.nn.init.uniform_ = skip
#     torch.nn.init.normal_ = skip
#     # from transformers import LlamaForCausalLM
#     tokenizer = AutoTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left', device_map="auto")

            
#     model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

#     quantizer = GPTQQuantizer(bits=args.wbits, dataset="c4", model_seqlen = 2048)
#     quantized_model = quantizer.quantize_model(model, tokenizer)

#     return quantized_model, quantizer


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--local-path', type=str, default=False,
        help='Local path of the model to quantize.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--autogptq', type=str, default=False,
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--gptqquantizer', type=str, default=False,
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--eval-from-path', action='store_true',
        help='Whether to evaluate the model from existing directory.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()
    if args.eval_from_path and os.path.exists(Path(args.save).name + "-gptq.pt"):
        print(f'Evaluating from checkpoint {Path(args.save).name + "-gptq.pt"}')

        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from transformers import LlamaTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 torch_dtype=torch.float16, 
                                                 device_map="auto",
                                                 offload_folder="./offload"
                                                 )
        # tokenizer = LlamaTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left')

        print('model', args.model)
        model.eval()

        seq_len = 2048
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=seq_len
        )

        datasets = ['wikitext2', 'ptb', 'c4'] 
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            if dataset == args.dataset:
                dataloader, testloader = get_loaders(
                    dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
                )
                print(dataset)
                alma_eval(model, testloader, DEV)

    else:
        if args.autogptq:
            print(f'Quantizing basemodel with autogptq')
            print('model', args.model)
            model = quantize_alma_autgptq(args.model)

            if args.save:
                output_path = Path(args.save).name + "-autogptq.pt"
                # save quantized model
                model.save_quantized(output_path)
                print(f"Saved autogptq smoothed model at {args.save}")

        if args.gptqquantizer:
            print(f'Quantizing basemodel with gptqquantizer')
            print('model', args.model)
            model, quantizer = quantize_gptqquantizer(args.model)

            if args.save:
                output_path = Path(args.save).name + "-gptq-quant.pt"
                # save quantized model
                quantizer.save(model, output_path)
                # model.save_quantized(output_path)
                print(f"Saved autogptq smoothed model at {args.save}")

        else:
            if args.local_path:
                print(f'Quantizing basemodel from {args.local_path}')
                print('with args.model', args.model)    
            else:
                print(f'Quantizing basemodel from {args.model}')
                print('with args.model', args.model)

            model = quantize_alma(args.model)
            

            if args.save:

                output_path = Path(args.save).name + "-gptq.pt"

                model.save_pretrained(output_path)
                print(f"Saved gptq smoothed model at {args.save}")

            
        # datasets = ['wikitext2', 'ptb', 'c4'] 
        # if args.new_eval:
        #     datasets = ['wikitext2', 'ptb-new', 'c4-new']
        # for dataset in datasets:
        #     if dataset == args.dataset:
        #         dataloader, testloader = get_loaders(
        #             dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        #         )
        #         print(dataset)
        #         alma_eval(model, testloader, DEV)

