import time

import torch
import torch.nn as nn
import os

from gptq import *
from modelutils import *
from quant import *
from pathlib import Path


def get_alma(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    # from transformers import LlamaForCausalLM
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import LlamaTokenizer

    model = AutoModelForCausalLM.from_pretrained(f"{args.model}-Pretrain", torch_dtype=torch.float16, device_map="auto")
    # model = PeftModel.from_pretrained(model, f"{args.model}-Pretrain-LoRA")
    # tokenizer = LlamaTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left')

    model.seqlen = 2048
    return model

@torch.no_grad()
def alma_sequential(model, dataloader, dev):
    print('Starting ALMA ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # print('model.model.model', model.model.model)
    print('model.model.model.embed_tokens', model.model.model.embed_tokens)
    layers = model.model.model.layers
    model.model.model.embed_tokens = model.model.model.embed_tokens.to(dev)
    model.model.model.norm = model.model.model.norm.to(dev)
   
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()
    model.model.model.norm = model.model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def alma_eval(model, testenc, dev):
    print('Evaluating ALMA ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.model.layers

    model.model.model.embed_tokens = model.model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.model.embed_tokens = model.model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.model.norm is not None:
        model.model.model.norm = model.model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.model.norm is not None:
            hidden_states = model.model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def alma_pack3(model, quantizers):
    layers = find_layers(model)
    ## probably quantizers is not a list and I am changing it to a list.
    ## so instead of doing this
    # quantizers = ['base_model.model.' + n for n in quantizers]
    # print('quantizers', quantizers)
    ## we  will do this
    quantizers = {f'base_model.model.{key}': value for key, value in quantizers.items()}

    # quantizers = ['base_model.model.' + n for n in quantizers]
    # print('PRINTED LIST', [n for n in quantizers])
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


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
        model.seqlen = 2048
        # tokenizer = LlamaTokenizer.from_pretrained(f"{args.model}-Pretrain", padding_side='left')

        print('model', args.model)
        model.eval()

        
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
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
        model = get_alma(args.model)
        print(f'Quantizing basemodel')
        print('model', args.model)
        
        model.eval()
        print('seq len', model.seqlen)

        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )

        if args.wbits < 16 and not args.nearest:
            tick = time.time()
            # if os.path.exists(args.save_quantizers):
            #     print('Loading quantizer from quantized checkpoint')
            #     pass
            #     ##to implement
            # else:
            quantizers = alma_sequential(model, dataloader, DEV)    
            
            print('Total time:', time.time() - tick)


        if args.save:

            model = alma_pack3(model, quantizers)
            
            # torch.save(model.state_dict(), args.save)
            # torch.save(quantizers, args.save_quantizers)
            # print(f'Model and quantizers saved at {args.save} and {args.save_quantizers}')
            output_path = Path(args.save).name + "-gptq.pt"

            model.save_pretrained(output_path)
            print(f"Saved smoothed model at {args.save}")

        
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

