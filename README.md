# Fast_SFT
![Fast_SFT](assets/title.png)

<p align="center">
   📃 <a href="https://eegb6fzscd.feishu.cn/wiki/XTMBwrMBgii6nnkqNsLcZHxznLE?from=from_copylink" target="_blank">Doc</a> 
</p>

- Fast LLM SFT CodeBase [Zero+Megatron+FlashAtten+compiler]; 
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
1. Complete "Prepare Model and Data"
2. Add 3.pretrain_baichuanxxx.sh

## Citation
```
@misc{fastsft,
  title={Fast LLM SFT CodeBase},
  author={Xidong Wang},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/wangxidong06/Fast_SFT}},
}
```
