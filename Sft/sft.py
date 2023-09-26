#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional
import torch, os, sys
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer

sys.path.append('../Lib/')
import data

tqdm.pandas()

lama_model = '7b-chat'
model_path = f'/home/dima/Models/Llama/Llama-2-{lama_model}-hf'
print('loading model:', model_path)

data_base_path = os.environ['DATA_ROOT']
drbench_train_path = 'DrBench/Csv/summ_0821_train.csv'
data_csv_path = os.path.join(data_base_path, drbench_train_path)

@dataclass
class ScriptArguments:
    """Parameters and their default settings"""

    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(
        default=32, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(
        default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(
        default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    output_dir: Optional[str] = field(
        default="Output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(
        default=5, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit)
    device_map = 'auto'
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype)

# todo: HF example only uses 'train' split; where is the validation data?
# dataset = load_dataset(script_args.dataset_name, split="train")
dataset = data.csv_to_fine_tune_data(data_csv_path)

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=False,
    hub_model_id=None)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules='q_proj,v_proj,gate_proj,up_proj,down_proj'.split(','),
        bias='none',
        task_type='CAUSAL_LM')
else:
    peft_config = None

trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config)

trainer.train()

trainer.save_model(script_args.output_dir)
