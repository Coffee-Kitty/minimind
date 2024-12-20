import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from model.model import Transformer
from model.LMConfig import LMConfig
warnings.filterwarnings('ignore')


def init_model():
    device = 'cuda:0'
    # Do model patching and add fast LoRA weights
    
    tokenizer_name_or_path = "./model/minimind_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    lm_config = LMConfig()
    model = Transformer(lm_config)
    # ckpt = "./minimind_dpo/dpo_512.pth"   
    ckpt = "./out/full_sft_512.pth"   
    state_dict = torch.load(ckpt,map_location=device)

    model.load_state_dict(state_dict, strict=False)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device)

    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = init_model()
    training_config = DPOConfig(
        output_dir="./minimind_dpo",
        per_device_train_batch_size=1,
        remove_unused_columns=False,
        report_to="none",
        save_steps=2000,
        learning_rate=4e-5,
        save_safetensors=False, # 设置为False，改为保存为pytorch格式的模型  
    )

    dataset_path = './dataset/dpo/dpo_demo.json'
    train_dataset = load_dataset('json', data_files=dataset_path)
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512
    )
    dpo_trainer.train()
    dpo_trainer.save_model("./minimind_dpo")

    # 7. save
    model = dpo_trainer.model
    ckp = f'./minimind_dpo/dpo_512.pth'

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, ckp)


