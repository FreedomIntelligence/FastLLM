# Fastest_SFT
Fastest LLM SFT CodeBase [Zero+Megatron+FlashAtten+compiler]; With dynamic strategy choosing

## How to use
1. Get Model size, GPU memory usage, training time and strategy.
2. Prepare Model and Data.
3. Use recommended parameters in 1. for training.
4. Calculate TFLOPs.

### 1. Calculate reference indicators
Modify the CONSTANT in pre_train_math.py
```
python 1.pre_train_math.py
```

### 2. Prepare Model and Data
Modify the CONSTANT in pre_train_math.py
```
python 2.py
```

### 3. Train Model
Modify the CONSTANT in pre_train_math.py
```
python 3.pretrain.py
```

### 4. Calculate TFLOPs.
Modify the CONSTANT in pre_train_math.py
```
python 4.aft_train_math.py
```