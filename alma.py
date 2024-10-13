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



# LANG_MAP = {
#     'en': 'English',
#     'de': 'German',
#     'cs': 'Czech',
#     'ru': 'Russian',
#     'zh': 'Chinese',
#     'is': 'Icelandic'
# }

# def get_alma(nsamples, seed, seqlen, tokenizer, source_lang, target_lang):
#     from datasets import Dataset
#     import pandas as pd
#     import random

#     # Define all language directions
#     full_alma_splits = {
#         'train': {
#             'cs-en': 'cs-en/train-00000-of-00001-3a60b130a713425b.parquet', # ok
#             'de-en': 'de-en/train-00000-of-00001-39460826cd7ac756.parquet', # ok 
#             # 'is-en': 'is-en/train-00000-of-00001-f71a989f63b28d68.parquet', # ok
#             'ru-en': 'ru-en/train-00000-of-00001-3ba3fad04eea46f0.parquet', # ok
#             'zh-en': 'zh-en/train-00000-of-00001-6bd744feceb30dbf.parquet'  # ok
#         },
#         'validation': {
#             'cs-en': 'cs-en/validation-00000-of-00001-d1f9a3fc339fbc84.parquet', # ok
#             'de-en': 'de-en/validation-00000-of-00001-34198d3f975c1787.parquet', # ok
#             # 'is-en': 'is-en/validation-00000-of-00001-bb3b8280f4b7ff31.parquet',
#             'ru-en': 'ru-en/validation-00000-of-00001-e9c97fe731036b74.parquet', # ok
#             'zh-en': 'zh-en/validation-00000-of-00001-d1cc83e30e3dcdb2.parquet'  # ok
#         }
#     }


#     LANG_MAP = {
#         'en': 'English',
#         'de': 'German',
#         'cs': 'Czech',
#         'ru': 'Russian',
#         'zh': 'Chinese',
#         'is': 'Icelandic'
#     }
   
#     # Set seed for reproducibility
#     random.seed(seed)

#     # Prepare the data loader
#     trainloader = []
#     all_validation_texts = []

#     # Iterate over all language directions
#     for lang_pair in full_alma_splits['train'].keys():
#         print(f"Loading dataset for {lang_pair}")

        # Load train and validation splits
#         train_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['train'][lang_pair]}"
#         print("Train link: ", train_split_path)
#         train_df = pd.read_parquet(train_split_path)

#         # Handle the 'is-en' case by splitting off validation data
#         if lang_pair == 'is-en':
#             print(f"Splitting 'is-en' training data into train and validation sets")
#             # Split 10% of the training data into validation
#             train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=seed)
#             valdata = Dataset.from_pandas(val_df)
#         else:
#             val_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['validation'][lang_pair]}"
#             print("Validation link: ", val_split_path)
#             val_df = pd.read_parquet(val_split_path)
#             valdata = Dataset.from_pandas(val_df)
            
#         # Convert DataFrames to Hugging Face Dataset
#         traindata = Dataset.from_pandas(train_df)
#         valdata = Dataset.from_pandas(val_df)

#         source_lang, target_lang = lang_pair.split('-')

#         # Sample data for training
#         for _ in range(nsamples):
#             while True:
#                 i = random.randint(0, len(traindata) - 1)
#                 translations = traindata[i]['translation']
#                 source_text = translations.get(source_lang)
#                 target_text = translations.get(target_lang)

#                 if source_text is None or target_text is None:
#                     continue  # Skip this entry if either source or target text is missing

#                 prompt = (
#                     f"Translate this from {LANG_MAP[source_lang]} to {LANG_MAP[target_lang]}:\n"
#                     f"{LANG_MAP[source_lang]}: {source_text}\n"
#                     f"{LANG_MAP[target_lang]}:"
#                 )

#                 if len(prompt.split()) <= seqlen:
#                     break

#                 # Append the prompt and target text as strings
#             trainloader.append((prompt, target_text))
#                 # # Tokenize source and target text
#                 # print('prompt', prompt)
#                 # print('target_text', target_text)

#                 # source_enc = tokenizer(prompt, return_tensors='pt', max_length=seqlen, truncation=True)
#                 # target_enc = tokenizer(target_text, return_tensors='pt', max_length=seqlen, truncation=True)
                
#                 # if source_enc.input_ids.shape[1] <= seqlen:
#                 #     break

#             # # Prepare the input and target sequences for training
#             # inp = source_enc.input_ids
#             # tar = target_enc.input_ids.clone()
#             # trainloader.append((inp, tar))
        
#         return trainloader

# def get_alma(nsamples, seed, seqlen, tokenizer, source_lang, target_lang):
#     from datasets import Dataset
#     import pandas as pd
#     import random
#     from sklearn.model_selection import train_test_split

#     # Define all language directions
#     full_alma_splits = {
#         'train': {
#             'cs-en': 'cs-en/train-00000-of-00001-3a60b130a713425b.parquet',
#             'de-en': 'de-en/train-00000-of-00001-39460826cd7ac756.parquet',
#             'ru-en': 'ru-en/train-00000-of-00001-3ba3fad04eea46f0.parquet',
#             'zh-en': 'zh-en/train-00000-of-00001-6bd744feceb30dbf.parquet'
#         },
#         'validation': {
#             'cs-en': 'cs-en/validation-00000-of-00001-d1f9a3fc339fbc84.parquet',
#             'de-en': 'de-en/validation-00000-of-00001-34198d3f975c1787.parquet',
#             'ru-en': 'ru-en/validation-00000-of-00001-e9c97fe731036b74.parquet',
#             'zh-en': 'zh-en/validation-00000-of-00001-d1cc83e30e3dcdb2.parquet'
#         }
#     }

#     LANG_MAP = {
#         'cs': 'Czech', 'de': 'German', 'en': 'English', 'ru': 'Russian', 'zh': 'Chinese'
#     }
   
#     # Set seed for reproducibility
#     random.seed(seed)

#     # Prepare the data loader
#     trainloader = []

#     # Iterate over all language directions
#     for lang_pair in full_alma_splits['train'].keys():
#         print(f"Loading dataset for {lang_pair}")

#         # Load train and validation splits
#         train_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['train'][lang_pair]}"
#         print("Train link: ", train_split_path)
#         train_df = pd.read_parquet(train_split_path)

#         # Handle the 'is-en' case by splitting off validation data
#         if lang_pair == 'is-en':
#             print(f"Splitting 'is-en' training data into train and validation sets")
#             # Split 10% of the training data into validation
#             train_df, _ = train_test_split(train_df, test_size=0.1, random_state=seed)
#         else:
#             val_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['validation'][lang_pair]}"
#             print("Validation link: ", val_split_path)
            
#         # Convert DataFrame to Hugging Face Dataset
#         traindata = Dataset.from_pandas(train_df)

#         source_lang, target_lang = lang_pair.split('-')

#         # Sample data for training
#         for _ in range(nsamples):
#             while True:
#                 i = random.randint(0, len(traindata) - 1)
#                 translations = traindata[i]['translation']
#                 source_text = translations.get(source_lang)
#                 target_text = translations.get(target_lang)

#                 if source_text is None or target_text is None:
#                     continue  # Skip this entry if either source or target text is missing

#                 prompt = (
#                     f"Translate this from {LANG_MAP[source_lang]} to {LANG_MAP[target_lang]}:\n"
#                     f"{LANG_MAP[source_lang]}: {source_text}\n"
#                     f"{LANG_MAP[target_lang]}:"
#                 )

#                 # Tokenize the prompt
#                 tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=seqlen, truncation=True)

#                 if tokenized_prompt.input_ids.shape[1] <= seqlen:
#                     break

#             # Prepare the input as a dictionary
#             example = {
#                 "input_ids": tokenized_prompt.input_ids.squeeze().tolist(),
#                 "attention_mask": tokenized_prompt.attention_mask.squeeze().tolist()
#             }
#             trainloader.append(example)
        
#     return trainloader


# def get_alma(nsamples, seed, seqlen, tokenizer, source_lang, target_lang):
#     # Define all language directions
#     full_alma_splits = {
#         'train': {
#             'cs-en': 'cs-en/train-00000-of-00001-3a60b130a713425b.parquet',
#             'de-en': 'de-en/train-00000-of-00001-39460826cd7ac756.parquet',
#             'ru-en': 'ru-en/train-00000-of-00001-3ba3fad04eea46f0.parquet',
#             'zh-en': 'zh-en/train-00000-of-00001-6bd744feceb30dbf.parquet'
#         },
#         'validation': {
#             'cs-en': 'cs-en/validation-00000-of-00001-d1f9a3fc339fbc84.parquet',
#             'de-en': 'de-en/validation-00000-of-00001-34198d3f975c1787.parquet',
#             'ru-en': 'ru-en/validation-00000-of-00001-e9c97fe731036b74.parquet',
#             'zh-en': 'zh-en/validation-00000-of-00001-d1cc83e30e3dcdb2.parquet'
#         }
#     }

#     LANG_MAP = {
#         'cs': 'Czech', 'de': 'German', 'en': 'English', 'ru': 'Russian', 'zh': 'Chinese'
#     }
   
#     # Set seed for reproducibility
#     random.seed(seed)

#     all_datasets = []

#     # Iterate over all language directions
#     for lang_pair in full_alma_splits['train'].keys():
#         print(f"Loading dataset for {lang_pair}")

#         # Load train split
#         train_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['train'][lang_pair]}"
#         print("Train link: ", train_split_path)
#         train_df = pd.read_parquet(train_split_path)

#         # Convert DataFrame to Hugging Face Dataset
#         traindata = Dataset.from_pandas(train_df)

#         source_lang, target_lang = lang_pair.split('-')

#         def preprocess_function(examples):
#             prompts = [
#                 f"Translate this from {LANG_MAP[source_lang]} to {LANG_MAP[target_lang]}:\n"
#                 f"{LANG_MAP[source_lang]}: {src}\n"
#                 f"{LANG_MAP[target_lang]}:"
#                 for src in examples['translation'][source_lang]
#             ]
            
#             model_inputs = tokenizer(prompts, max_length=seqlen, truncation=True, padding="max_length")
            
#             return model_inputs

#         # Preprocess the dataset
#         tokenized_dataset = traindata.map(preprocess_function, batched=True)

#         # Sample from the dataset
#         sampled_dataset = tokenized_dataset.shuffle(seed=seed).select(range(min(nsamples, len(tokenized_dataset))))

#         all_datasets.append(sampled_dataset)

#     # Combine all datasets
#     final_dataset = concatenate_datasets(all_datasets)

#     return final_dataset

from typing import List
import pandas as pd
import random
from datasets import Dataset

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

