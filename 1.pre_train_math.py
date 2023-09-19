# doc: https://eegb6fzscd.feishu.cn/wiki/WpjTw04S1iVoVekU1vOci4Wcnhb?from=from_copylink
from tabulate import tabulate
import math

# Baichuan2-7B: NHIDDEN=4096 NLAYERS=32 SEQ_LEN=4096 VOCAB_SIZE=125696
# Baichaun2-13B: NHIDDEN=5120 NLAYERS=40 SEQ_LEN=4096 VOCAB_SIZE=125696
# Llama2-7B: NHIDDEN=4096 NLAYERS=32 SEQ_LEN=4096 VOCAB_SIZE=32000
# Llama2-13B: NHIDDEN=5120 NLAYERS=40 SEQ_LEN=4096 VOCAB_SIZE=32000
# Llama2-70B: NHIDDEN=8192 NLAYERS=80 SEQ_LEN=4096 VOCAB_SIZE=32000
NHIDDEN=4096
NLAYERS=32
SEQ_LEN=4096
VOCAB_SIZE=125696

NODE=1
GPU_PER_NODE=4
GPU_MEMORY=80
BATCH_SIZE=8
BTOKEN=0.7   # Token in Billion
TFLOPS=140   # [130-170]
Gradient_checkpointing=True # gradient_checkpointing technich: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9


NGPU=GPU_PER_NODE*NODE
h=NHIDDEN
l=NLAYERS
s=SEQ_LEN 
v=VOCAB_SIZE
b=BATCH_SIZE

def next_power_of_2(n):
    return 2 ** math.ceil(math.log2(n))

def main():
    print('-----------Model_Size and GPU_Mem-----------')
    emb=(v*h+s*h)/10**9
    blk=(12*h**2+6*h)/10**9
    extra_norm=h/10**9
    model=l*blk+emb+extra_norm
    single_mem=GPU_MEMORY-1

    dict={"Model size/B": round(model, 2), "ratio(NHIDDEN/NLAYERS)":int(h/l), "Usable_mem_per_GPU/G": round(single_mem, 2)}
    print(tabulate([dict], headers="keys", tablefmt="pretty"))


    print('-----------With Mixed Precision(bp16)-----------')
    print(f'-----Memory_reference_indicator(Batch_size={b})-----')
    input=(b*s*h)/10**9
    activation=math.sqrt((b*s*h*34*l)/10**9) if Gradient_checkpointing else (b*s*h*34*l)/10**9
    activation_b1=math.sqrt((s*h*34*l)/10**9) if Gradient_checkpointing else (s*h*34*l)/10**9
    input_all=input+activation
    train_memory_factor=2+2+4*3
    total_memory=round(model*train_memory_factor+input_all*2, 2)
    list_of_dicts=[
        {'Module': 'emb', 'Size/B':round(emb, 2), 'Eval_memory/GB': round(emb*2, 2), 'Train_momery/GB': round(emb*train_memory_factor, 2)},
        {'Module': 'one_layer', 'Size/B':round(blk, 2), 'Eval_memory/GB': round(blk*2, 2), 'Train_momery/GB': round(blk*train_memory_factor, 2)},
        {'Module': 'input', 'Size/B':round(input, 2), 'Eval_memory/GB': round(input*2, 2), 'Train_momery/GB': round(input*2, 2)},
        {'Module': 'activation(batchsize=1)', 'Size/B':round(activation_b1, 2), 'Eval_memory/GB': round(activation_b1*2, 2), 'Train_momery/GB': round(activation_b1*2, 2)},
        {'Module': 'ALL', 'Size/B':round(model+input_all, 2), 'Eval_memory/GB': round(model*2+input_all*2, 2), 'Train_momery/GB': total_memory}
    ]
    print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))

    print(f'-----Strategy_reference_indicator(Batch_size={b})-----')
 
    train_memory_factor_zero1=2+2+(4*3)/NGPU
    train_memory_factor_zero2=2+(2+4*3)/NGPU
    train_memory_factor_zero3=(2+2+4*3)/NGPU

    list_of_dicts=[
        {'Strategy': 'Zero1','Eval_memory_per_gpu/GB': round(model*2, 2), 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero1+input_all*2, 2)},
        {'Strategy': 'Zero2','Eval_memory_per_gpu/GB': round(model*2, 2), 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero2+input_all*2, 2)},
        {'Strategy': 'Zero3','Eval_memory_per_gpu/GB': round(model*2/NGPU, 2), 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero3+input_all*2, 2)},
    ]
    print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))

    print(f'---------------------Strategy_Recommand---------------------')
    trianing_days=round(BTOKEN*1e9*8*model*1e9/(NGPU*TFLOPS*1e12*60*60*24),2) # https://arxiv.org/pdf/2104.04473.pdf 
    
    if total_memory>single_mem*NGPU:
        print(f'Minimal_Memory_needed:{total_memory}GB, Give_usable_memory:{single_mem}*{NGPU}={single_mem*NGPU}GB')
        print("You may try ZeRO-Infinity(Zero3+offload_param(nvme)+offload_optimizer(nvme)) strategy, If it doesn't work, please increase GPU. I'm sorry.")
        return
    
    if abs(list_of_dicts[0]['Train_momery_per_gpu/GB']-single_mem)<3 or list_of_dicts[0]['Train_momery_per_gpu/GB']<single_mem:
        if list_of_dicts[0]['Train_momery_per_gpu/GB']<single_mem:
            print('Recommand_Strategy:')
            list_of_dicts=[
                {'Zero': 'Zero1','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero1+input_all*2, 2), 'Trianing_days': trianing_days},
            ]
            print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
            print('Please find the best batch_size by adjusting BATCH_SIZE')
            return
        else:
            print('Recommand_Strategy:')
            list_of_dicts=[
                {'Zero': 'Zero1+offload','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero1+input_all*2, 2), 'Trianing_days': trianing_days},
            ]
            print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
            print('Please find the best batch_size by adjusting BATCH_SIZE')
            return
    elif list_of_dicts[1]['Train_momery_per_gpu/GB']<single_mem:
        print('Recommand_Strategy:')
        list_of_dicts=[
            {'Zero': 'Zero2','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero2+input_all*2, 2), 'Trianing_days': trianing_days},
        ]
        print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
        print('Please find the best batch_size by adjusting BATCH_SIZE')
        return
    
    print("You can't use pure Zero1 or Zero2 strategy.")
    
    single_node_mem=GPU_PER_NODE*single_mem
    if total_memory < single_node_mem:
        TP=GPU_PER_NODE
        DP=NGPU/TP
        train_memory_factor_zero1=2+2+(4*3)/DP
        print('Recommand_Strategy:')
        list_of_dicts=[
            {'Zero': 'Zero1+TP','DP': DP, 'TP': TP, 'PP':1, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero1/TP+input_all*2/TP, 2), 'Trianing_days': trianing_days},
            {'Zero': 'Zero3+(offload)','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero3+input_all*2, 2), 'Trianing_days': trianing_days},
        ]
        print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
        print('Please find the best batch_size by adjusting BATCH_SIZE')
        return
    else:
        PP=math.ceil(total_memory/single_node_mem)
        TP=GPU_PER_NODE
        DP=1 if NGPU/TP/PP < 1 else int(NGPU/TP/PP)
        if DP==1:
            PP=int(NGPU/TP)
        train_memory_factor_zero1=2+2+(4*3)/DP
        print('Recommand_Strategy:')
        list_of_dicts=[
            {'Zero': 'Zero1+TP+PP','DP': DP, 'TP': TP, 'PP':PP, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero1/PP/TP+input_all*2/TP, 2), 'Trianing_days': trianing_days},
            {'Zero': 'Zero3+(offload)','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(model*train_memory_factor_zero3+input_all*2, 2), 'Trianing_days': trianing_days},
        ]
        print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
        print('Please find the best batch_size by adjusting BATCH_SIZE')
        return
        


 

if __name__=="__main__":
    main()




