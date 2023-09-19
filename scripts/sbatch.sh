#!/bin/bash
#SBATCH -J test
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --cpus-per-task=96
#SBATCH --reservation=root_114  # 仅限于wangbeny用户 其余用户不用这一行
#SBATCH -w pgpu17
#SBATCH --gres=gpu:8

bash /mntcephfs/data/med/xidong/yaojishi/gen_ans.sh   # 要运行的命令行指令