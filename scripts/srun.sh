#!/bin/bash 
srun --job-name=test  --gres=gpu:2   -w pgpu20 -p p-A100 -c 24  --reservation=root_114 --pty bash 