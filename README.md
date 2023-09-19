# Fast-LLM
![Fast_SFT](assets/title.png)

<p align="center">
   📃 <a href="https://eegb6fzscd.feishu.cn/wiki/XTMBwrMBgii6nnkqNsLcZHxznLE?from=from_copylink" target="_blank">Doc</a> 
</p>

- Fast LLM Training CodeBase [Deepspeed+Megatron+FlashAttention+CudaFusionKernel+Compiler]
- With dynamic strategy choosing

## How to use
1. Get Model size, GPU memory usage, training time and strategy.
2. Prepare Model and Data.
3. Use recommended parameters in 1. for training.
4. Calculate TFLOPs.
5. Transfer model to Huggingface format. 

### 1. Calculate reference indicators
Modify the CONSTANT in pre_train_math.py
```
python 1.pre_train_math.py
```
The output of training LLAMA-70B with 6Node*4GPU(80G) and 0.7B tokens:
```
-----------Model_Size and GPU_Mem-----------
+--------------+------------------------+----------------------+
| Model size/B | ratio(NHIDDEN/NLAYERS) | Usable_mem_per_GPU/G |
+--------------+------------------------+----------------------+
|    64.72     |          102           |          79          |
+--------------+------------------------+----------------------+
-----------With Mixed Precision(bp16)-----------
-----Memory_reference_indicator(Batch_size=8)-----
+-------------------------+----------+------------------+-------------------+
| Module                  |   Size/B |   Eval_memory/GB |   Train_momery/GB |
+=========================+==========+==================+===================+
| emb                     |     0.3  |             0.59 |              4.73 |
+-------------------------+----------+------------------+-------------------+
| one_layer               |     0.81 |             1.61 |             12.89 |
+-------------------------+----------+------------------+-------------------+
| input                   |     0.27 |             0.54 |              0.54 |
+-------------------------+----------+------------------+-------------------+
| activation(batchsize=1) |     9.55 |            19.11 |             19.11 |
+-------------------------+----------+------------------+-------------------+
| ALL                     |    92.01 |           184.03 |           1090.17 |
+-------------------------+----------+------------------+-------------------+
-----Strategy_reference_indicator(Batch_size=8)-----
+------------+--------------------------+---------------------------+
| Strategy   |   Eval_memory_per_gpu/GB |   Train_momery_per_gpu/GB |
+============+==========================+===========================+
| Zero1      |                   129.45 |                    345.84 |
+------------+--------------------------+---------------------------+
| Zero2      |                   129.45 |                    221.78 |
+------------+--------------------------+---------------------------+
| Zero3      |                     5.39 |                     97.73 |
+------------+--------------------------+---------------------------+
---------------------Strategy_Recommand---------------------
You can't use pure Zero1 or Zero2 strategy.
Recommand_Strategy:
+-----------------+------+------+------+---------------------------+-----------------+
| Zero            |   DP |   TP |   PP |   Train_momery_per_gpu/GB |   Trianing_days |
+=================+======+======+======+===========================+=================+
| Zero1+TP+PP     |    1 |    4 |    6 |                     56.79 |            1.25 |
+-----------------+------+------+------+---------------------------+-----------------+
| Zero3+(offload) |   24 |    1 |    1 |                     97.73 |            1.25 |
+-----------------+------+------+------+---------------------------+-----------------+
Please find the best batch_size by adjusting BATCH_SIZE
```

### 2. Prepare Model and Data
Under Construction
#### 2.1 Prepare Model
- LLAMA:
  1. Convert LLAMA from Meta format checkpoints to HF format
    ```
    python /src/tools/convert_checkpoint/convert_llama_weights_to_hf.py --input_dir $LLAMA_FORMAT_DIR --output_dir $HF_FORMAT_DIR --model_size 7B
    # --model_size include 7B, 13B, and 70B (for pretrained-only models), and 7Bf, 13Bf, and 70Bf (for chat-finetuned models).
    ```

  2. Convert HF checkpoints to Megatron format
    ```
    python /src/tools/checkpoint/util.py \
          --model-type GPT \
          --loader llama2_hf \
          --saver megatron \
          --target-tensor-parallel-size ${TP} \
          --load-dir ${HF_FORMAT_DIR} \
          --save-dir ${MEGATRON_FORMAT_DIR} \
          --tokenizer-model ${TOKENIZER_MODEL}
    ```
- Others:
```
python tools/convert_checkpoint/deepspeed_to_megatron.py --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER --target_tp TARGET_TP --target_pp TARGET_PP 
```
#### 2.2 Prepare Data
Data_item is in jsonl
```
{"text": "The quick brown fox"}
{"text": "jumps over the lazy dog"}
```
The name of the text field of the json can be changed by using the --json-key flag in preprocess_data.py. "text" by default

```
python tools/preprocess_data.py \
       --input data.json \
       --output-prefix llama2 \
       --vocab-file VOCAB_FILE \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
```


### 3. Train Model
Modify the CONSTANT in 3.pretrain_xxxxxx.sh
```
bash 3.pretrain_xxxxxx.sh
```

### 4. Calculate TFLOPs.
Modify the CONSTANT in aft_train_math.py
```
python 4.aft_train_math.py
```

### 5. Transfer model to HF Transformers. 

```bash
python /src/tools/convert_checkpoint/deepspeed_to_transformers.py  \
--input_folder /path/to/checkpoint \
--output_folder /path/to/transformers/checkpoint
```

## To do list
1. Support Baichuan2.
2. Support Instruction tuning.
3. Benchmark TFLOPS with other Repo on different settings

## Acknowledgement
- Megatron-DeepSpeed: https://github.com/microsoft/Megatron-DeepSpeed
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM



## Citation
```
@misc{fastllm,
  title={Fast LLM Training CodeBase},
  author={Xidong Wang},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wangxidong06/Fast_LLM}},
}
```
