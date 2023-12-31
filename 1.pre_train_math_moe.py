# doc: https://docs.google.com/presentation/d/1FlkJ98hSGnZ4kjePaxCJvNkM9zvlkYKIZ6xu8TjJGXE/edit?usp=sharing
from tabulate import tabulate
import math

# Mixtraol-8*7B
NHIDDEN=4096
NLAYERS=32
NHEAD=8
SEQ_LEN=4096
VOCAB_SIZE=32000
EXPERT_NUM=8
EXPERT_NUM_LIVE=2

NODE=1
GPU_PER_NODE=8
GPU_MEMORY=80
BATCH_SIZE=1
BTOKEN=0.7   # Token in Billion
TFLOPS=140   # [130-170]
Gradient_checkpointing=True # gradient_checkpointing technich: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9


NGPU=GPU_PER_NODE*NODE
h=NHIDDEN
l=NLAYERS
s=SEQ_LEN 
v=VOCAB_SIZE
b=BATCH_SIZE
a=NHEAD
expert_num=EXPERT_NUM
expert_num_live=EXPERT_NUM_LIVE
batch_expert_num_dict={'1':2,'2':4,'4':6, '8':8}
expert_num_live_batch=batch_expert_num_dict[str(expert_num)]

def main():
    print('-----------Model_Size and GPU_Mem-----------')
    emb=(v*h+s*h)/10**9
    mlp=8*h**2
    rounting=expert_num*h
    attn_and_norm=4*h**2+6*h
    blk=(mlp*expert_num+attn_and_norm+rounting)/10**9
    blk_train=(mlp*expert_num_live_batch+attn_and_norm+rounting)/10**9
    extra_norm=h/10**9
    model=l*blk+emb+extra_norm
    model_train=l*blk_train+emb+extra_norm
    single_mem=GPU_MEMORY-1

    dict={"Model size/B": round(model, 2), "ratio(NHIDDEN/NLAYERS)":int(h/l), "Usable_mem_per_GPU/G": round(single_mem, 2)}
    print(tabulate([dict], headers="keys", tablefmt="pretty"))


    print('-----------With Mixed Precision(bp16)-----------')
    print(f'-----Memory_reference_indicator(Batch_size={b})-----')
    input=(b*s*h)/10**9
    activation_per_MLP = 19*s*b*h
    activation_per_layer = b*s*h*15 + activation_per_MLP*expert_num_live_batch +5*s*s*b*a
    activation=math.sqrt((activation_per_layer*l)/10**9) if Gradient_checkpointing else (activation_per_layer*l)/10**9
    activation_b1=math.sqrt((activation_per_layer*l/b)/10**9) if Gradient_checkpointing else (activation_per_layer*l/b)/10**9
    input_all=input+activation
    train_memory_factor=2+2+4*3
    train_memory = 2*model + (2+4*3)*model_train
    total_memory=round(train_memory+input_all*2, 2)
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
    train_memory_zero1=2*model+(2+(4*3))/NGPU*model_train
    train_memory_factor_zero2=2+(2+4*3)/NGPU
    train_memory_zero2=2*model+(2+4*3)/NGPU*model_train
    train_memory_factor_zero3=(2+2+4*3)/NGPU
    train_memory_zero3=2*model/NGPU+(2+(4*3))/NGPU*model_train

    list_of_dicts=[
        {'Strategy': 'Zero1','Eval_memory_per_gpu/GB': round(model*2, 2), 'Train_momery_per_gpu/GB': round(train_memory_zero1+input_all*2, 2)},
        {'Strategy': 'Zero2','Eval_memory_per_gpu/GB': round(model*2, 2), 'Train_momery_per_gpu/GB': round(train_memory_zero2+input_all*2, 2)},
        {'Strategy': 'Zero3','Eval_memory_per_gpu/GB': round(model*2/NGPU, 2), 'Train_momery_per_gpu/GB': round(train_memory_zero3+input_all*2, 2)},
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
                {'Zero': 'Zero1','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(train_memory_zero1+input_all*2, 2), 'Trianing_days': trianing_days},
            ]
            print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
            print('Please find the best batch_size by adjusting BATCH_SIZE')
            return
        else:
            print('Recommand_Strategy:')
            list_of_dicts=[
                {'Zero': 'Zero1+offload','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(train_memory_zero1+input_all*2, 2), 'Trianing_days': trianing_days},
            ]
            print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
            print('Please find the best batch_size by adjusting BATCH_SIZE')
            return
    elif list_of_dicts[1]['Train_momery_per_gpu/GB']<single_mem:
        print('Recommand_Strategy:')
        list_of_dicts=[
            {'Zero': 'Zero2','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(train_memory_zero2+input_all*2, 2), 'Trianing_days': trianing_days},
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
            {'Zero': 'Zero1+TP','DP': DP, 'TP': TP, 'PP':1, 'Train_momery_per_gpu/GB': round(train_memory_zero1/TP+input_all*2/TP, 2), 'Trianing_days': trianing_days},
            {'Zero': 'Zero3+(offload)','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(train_memory_zero3+input_all*2, 2), 'Trianing_days': trianing_days},
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
            {'Zero': 'Zero1+TP+PP','DP': DP, 'TP': TP, 'PP':PP, 'Train_momery_per_gpu/GB': round(train_memory_zero1/PP/TP+input_all*2/TP, 2), 'Trianing_days': trianing_days},
            {'Zero': 'Zero3+(offload)','DP': NGPU, 'TP': 1, 'PP':1, 'Train_momery_per_gpu/GB': round(train_memory_zero3+input_all*2, 2), 'Trianing_days': trianing_days},
        ]
        print(tabulate(list_of_dicts, headers="keys", tablefmt="grid"))
        print('Please find the best batch_size by adjusting BATCH_SIZE')
        return
        


 

if __name__=="__main__":
    main()




