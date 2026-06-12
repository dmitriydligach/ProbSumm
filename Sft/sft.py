#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional
import torch, os, sys
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from trl import SFTConfig, SFTTrainer

sys.path.append('../Lib/')
import data

tqdm.pandas()

@dataclass
class ScriptArguments:
    """Parameters and their default settings"""

    model_name: Optional[str] = field(
        default="/home1/shared/Models/Llama-3.2-1B-Instruct",
        metadata={"help": "HF hub model ID or local path"})
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(
        default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(
        default=2, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(
        default=True, metadata={"help": "load the model in 4 bits precision (QLoRA)"})
    use_peft: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use PEFT or not to train adapters"})
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

print('loading model:', script_args.model_name)

local_rank = int(os.environ.get('LOCAL_RANK', -1))
device_map = 'auto' if local_rank == -1 else {'': local_rank}

if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    torch_dtype = torch.bfloat16
elif script_args.load_in_8bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    torch_dtype = torch.bfloat16
else:
    quantization_config = None
    torch_dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
tokenizer.model_max_length = script_args.seq_length

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    dtype=torch_dtype)
model.config.use_cache = False

data_base_path = os.environ['DATA_ROOT']
train_csv_path = 'ProbSumm/BioNLP2023-1A-Train.csv'
data_csv_path = os.path.join(data_base_path, train_csv_path)

dataset = data.csv_to_llama3_chat_format(data_csv_path, tokenizer)

training_args = SFTConfig(
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
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    push_to_hub=False,
    hub_model_id=None,
    dataset_text_field=script_args.dataset_text_field)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'.split(','),
        bias='none',
        task_type='CAUSAL_LM')
else:
    peft_config = None

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config)

trainer.train()

trainer.save_model(script_args.output_dir)