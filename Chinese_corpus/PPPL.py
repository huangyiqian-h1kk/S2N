import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from transformers import set_seed
from transformers import BertTokenizerFast
from transformers import TFAutoModelForMaskedLM
from transformers import BertForPreTraining, BertForMaskedLM, BertConfig
import pandas as pd
import numpy as np
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import argparse, torch, datasets, ast, operator, gc, os
import torch.nn as nn
import json
import csv

from torch.nn import CrossEntropyLoss
import time

from torch.nn import CrossEntropyLoss
import time

def batched_PPPL(model, tokenizer, batch):
    
    torch.cuda.empty_cache()
    time1=time.time()
    encoding = tokenizer(this_batch, return_tensors='pt', padding=True, truncation=True, max_length=80, return_overflowing_tokens=True, stride=6)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    max_length = input_ids.shape[1]
    # 创建掩码并构造标签
    mask1 = (torch.ones(max_length-1).diag(1)==1).to(device)
    repeat_input = input_ids.unsqueeze(1) .repeat(1, max_length, 1)

    #加一个pad_mask跟mask1相乘
    pad_tensor= (torch.ones(max_length)*tokenizer.pad_token_id).repeat(max_length,1).to(device)
    #filter_tensor= (torch.zeros(max_length)*tokenizer.pad_token_id).repeat(max_length,1).to(device)

    pad_mask= repeat_input!=pad_tensor
    pad_pos=torch.unique(input=pad_mask, dim=1, return_inverse=False, sorted=False).squeeze(1)
    avail_mask= (pad_pos.unsqueeze(1))*(pad_pos.unsqueeze(-1))

    masked_input = torch.where(mask1, tokenizer.mask_token_id, repeat_input)
    masked_input *= avail_mask

    seq_pad_filter = (((masked_input==tokenizer.mask_token_id).any(dim=-1))*((masked_input==tokenizer.sep_token_id).any(dim=-1))).unsqueeze(-1)
    masked_input*=seq_pad_filter

    labels = torch.where(mask1, input_ids.unsqueeze(1), torch.tensor(-100).to(device))
    #print()
    masked_input = masked_input.view(-1, max_length)

    labels = labels.view(-1, max_length)

    attention_mask = attention_mask.unsqueeze(1).repeat(1, max_length, 1).view(-1, max_length)
    
    
    
    with torch.inference_mode(mode=True):
        output = model(masked_input, attention_mask=attention_mask, labels=labels).logits
        loss_fct = CrossEntropyLoss(reduction="none") 
        
        del input_ids, mask1, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
    
        if torch.cuda.device_count() > 1:
            loss = loss_fct(output.view(-1, model.module.config.vocab_size), labels.view(-1))
        else:
            loss = loss_fct(output.view(-1, model.config.vocab_size), labels.view(-1))
        loss = loss.view(pad_mask.shape)*seq_pad_filter
        batched_loss = torch.sum(torch.sum(loss,dim=1),dim=1)/(torch.count_nonzero(seq_pad_filter,dim=1).squeeze())

    time2=time.time()
    print("time:",(time2-time1))
    # 计算困惑度
    ppl = torch.exp(batched_loss)
    
    #del input_ids, attention_mask, mask1
    del repeat_input,pad_tensor, pad_mask, output, labels,batched_loss,loss
    gc.collect()
    
    torch.cuda.empty_cache()
    
    PPL=ppl.cpu().numpy()
    o_m=encoding['overflow_to_sample_mapping'].tolist()
    PPPLs=list()
    
    for i in range(len(set(o_m))):
        if o_m.count(i)>1:
            pos_ppl=[PPL[j]*(o_m[j]==i) for j in range(len(PPL))]
            pos_ppl=[i for i in pos_ppl if not i==0]
            mean_ppl= sum(pos_ppl[:-1])/(len(pos_ppl)-1)

        else:
            pos_ppl=[PPL[j]*(o_m[j]==i) for j in range(len(PPL))]
            pos_ppl=[i for i in pos_ppl if not i==0]
            mean_ppl= pos_ppl
        PPPLs.append(mean_ppl)
    
    return PPPLs

parser = argparse.ArgumentParser(description='search for best template according to dev set')
parser.add_argument('--max_len', default=512, type=int, help="max sequence length")
parser.add_argument('--model', default='/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning//MLM_task_tuned_models_for _datasets_PPL/MLM_tuned_roberta_large_JY_te+JY_Tr+CSI_tr/', type=str, help="pretrained model")
#parser.add_argument('--model', default='./chinese_roberta_wwm_large_ext_pytorch/')
parser.add_argument('--result_output_dir', default='/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning/Tune_on_CSI/csi_output/ppl_results/', type=str, help="output directory")
parser.add_argument('--tokenizer', default='/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning/MLM_task_tuned_models_for _datasets_PPL/MLM_tuned_roberta_large_JY_te+JY_Tr+CSI_tr/', type=str, help="tokenizer")
args = parser.parse_args(args=[])

# 加载模型和分词器
model = BertForMaskedLM.from_pretrained(args.model)
tokenizer = BertTokenizerFast.from_pretrained(f'{args.tokenizer}')
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)
train_file='/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning/Chinese_corpus/CPED6_enlarged_ratio_test.csv'
datasets = load_dataset("csv", data_files={"train": train_file}, sep=",")
train = datasets['train']["text"]
dataloader = DataLoader(train, batch_size=8, collate_fn=lambda x: x)

ppls_mean=list()
texts=list()
# 计算每个批次的 PPL
c=0

for this_batch in tqdm(dataloader):
    this_ppl= batched_PPPL(model, tokenizer,this_batch)
    ppls_mean+=this_ppl
    texts+=this_batch
            
 
    if c%50==0:
        ppls_csi_train_v1={"pppl":ppls_mean, "texts":texts}#将列表a，b转换成字典
        ppls_csi_train_v1=pd.DataFrame(ppls_csi_train_v1)
        ppls_csi_train_v1.to_csv('/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning/Chinese_corpus/CPED6_enlarged_ratio_test_PPPL.csv')


ppls_csi_train_v1={"pppl":ppls_mean, "texts":texts}#将列表a，b转换成字典
ppls_csi_train_v1=pd.DataFrame(ppls_csi_train_v1)
ppls_csi_train_v1.to_csv('/home/lr/h1k/KNN-BERT-main/csi_sequetial_fine_tuning/Chinese_corpus/CPED6_enlarged_ratio_test_PPPL.csv')
