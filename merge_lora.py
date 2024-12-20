import os
import platform
import argparse
import time
import math
import warnings
import torch
import pandas as pd
import torch.nn.functional as F
from contextlib import nullcontext

from torch import optim
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from torch.utils.data import DataLoader
from model.LMConfig import LMConfig
from model.dataset import SFTDataset
from model.model import Transformer


parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="Device to use")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="Weights & Biases project name")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
parser.add_argument("--warmup_iters", type=int, default=1000, help="Number of warmup iterations")
parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")

args = parser.parse_args()

lm_config = LMConfig()
max_seq_len = lm_config.max_seq_len
args.save_dir = os.path.join(args.out_dir)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)
tokens_per_iter = args.batch_size * max_seq_len
torch.manual_seed(1337)
device_type = "cuda" if "cuda" in args.device else "cpu"


tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')


model = Transformer(lm_config)
moe_path = '_moe' if lm_config.use_moe else ''
ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
state_dict = torch.load(ckp, map_location=args.device)
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

 
model = PeftModel.from_pretrained(model, args.save_dir)
# 合并并卸载 LoRA 权重
model = model.merge_and_unload()

ckp = f'{args.save_dir}/lora_sft{lm_config.dim}{moe_path}.pth'

if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()

torch.save(state_dict, ckp)
