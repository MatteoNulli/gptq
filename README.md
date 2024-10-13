# GPTQ

This repository contains the code for quantizing [ALMA](https://github.com/fe1ixxu/ALMA) using the GPTQ technique. This is forked from the original [GPTQ](https://github.com/IST-DASLab/gptq) repository. You can find the orginal README at `README_original.md`.

## Reproducibility

To reproduce our results run `scripts/alma.job` file. 

- `$HF_FOLDER`: change to choose a different ALMA version to quantize (ALMA-7B, ALMA-13B)
- `--wbits`: modify the bit quantization level of weights
- `--save`: change the save directory name

<!-- 
Here is a summary of ALMA results:

| Wiki2 PPL | FP16 | 4bit-RTN | 4bit-GPTQ | 3bit-RTN | 3bit-GPTQ | 3g128-GPTQ |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| LLaMa-7B  | 5.68 | 6.29     | **6.09**  | 25.54    | **8.07**  | 6.61       |
| LLaMa-13B | 5.09 | 5.53     | **5.36**  | 11.40    | **6.63**  | 5.62       |
| LLaMa-30B | 4.10 | 4.54     | **4.45**  | 14.89    | **5.69**  | 4.80       |
| LLaMa-65B | 3.53 | 3.92     | **3.84**  | 10.59    | **5.04**  | 4.17       |
 -->
<!-- 
Here is a sample command:

```
python llama.py LLAMA_HF_FOLDER c4 --wbits 4 --true-sequential --act-order --new-eval
```

The `--act-order` heuristic also dramatically improves accuracy on the OPT-66B outlier model: 9.55 to 9.34 and 14.16 to 9.95 PPL on Wiki2 for 4bit and 3bit, respectively. -->

## Dependencies

Dependencies are listed in `scripts/env_create.job` file and in `requirements.txt`.

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.
