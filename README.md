# Fastest_SFT
Fastest LLM SFT CodeBase [Zero+Megatron+FlashAtten+compiler]; With dynamic strategy choosing

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