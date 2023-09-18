model_size_in_B=7
seqlen=4096
global_batch_size=128
time_in_sec_per_interation=6.8
total_gpus=32

TFLOPS=model_size_in_B * 4 * 2 * seqlen * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3) # https://arxiv.org/pdf/2104.04473.pdf 
print(f'TFLOPS:{TFLOPS:.2f}')