import re, os
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from probsumm_utils import (load_config, load_dataset_from_csv, make_conversation,
                             extract_answer, normalize)

cfg = load_config()

data_base_path = os.environ['DATA_ROOT']
train_csv_path = os.path.join(data_base_path, cfg['data']['train_path'])

print('Loading training data...')
train_dataset = load_dataset_from_csv(train_csv_path)

local_rank = int(os.environ.get('LOCAL_RANK', -1))
device_map = 'auto' if local_rank == -1 else {'': local_rank}

tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], clean_up_tokenization_spaces=False)

max_prompt_length = cfg['training']['max_prompt_length']

def truncate_prompt(example):
    """Truncate input_text to max_prompt_length tokens before conversation formatting."""
    tokens = tokenizer(example['input_text'], truncation=True, max_length=max_prompt_length)
    example['input_text'] = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
    return example

train_dataset = train_dataset.map(truncate_prompt)
train_dataset = train_dataset.map(make_conversation)

print(f'Training examples: {len(train_dataset)}')

print(f'Loading model: {cfg["model_id"]}...')
model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], torch_dtype='auto', device_map=device_map)

lora_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=cfg['lora']['r'],
    lora_alpha=cfg['lora']['lora_alpha'],
    lora_dropout=cfg['lora']['lora_dropout'],
    target_modules=cfg['lora']['target_modules'])

print('Applying LoRA...')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def format_reward(completions, **kwargs):
    """Reward 1.0 if the completion matches <think>...</think><answer>...</answer>."""

    pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
    contents = [completion[0]['content'] for completion in completions]
    return [1.0 if re.match(pattern, c, re.DOTALL) else 0.0 for c in contents]


def problem_coverage_reward(completions, **kwargs):
    """F1 reward over exact-matched problems — penalizes both missed and extra problems (0–1)."""

    answers = kwargs['answer']
    contents = [completion[0]['content'] for completion in completions]
    rewards = []

    for content, reference in zip(contents, answers):
        generated_problems = set(normalize(p) for p in extract_answer(content).split(';') if p.strip())
        reference_problems = set(normalize(p) for p in reference.split(';') if p.strip())

        if not generated_problems or not reference_problems:
            rewards.append(0.0)
            continue

        tp = len(generated_problems & reference_problems)
        precision = tp / len(generated_problems)
        recall = tp / len(reference_problems)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rewards.append(f1)

    return rewards


training_args = GRPOConfig(
    output_dir=cfg['output_dir'],
    learning_rate=cfg['training']['learning_rate'],
    remove_unused_columns=False,  # keep 'answer' column for problem_coverage_reward
    gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
    num_train_epochs=cfg['training']['num_train_epochs'],
    bf16=cfg['training']['bf16'],
    max_completion_length=cfg['training']['max_completion_length'],
    num_generations=cfg['training']['num_generations'],
    report_to=['tensorboard'],
    logging_steps=cfg['training']['logging_steps'],
    push_to_hub=False,
    save_strategy='steps',
    save_steps=cfg['training']['save_steps'])

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, problem_coverage_reward],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer)

print('Starting GRPO training...')
trainer.train()

print(f'Saving model to {cfg["output_dir"]}...')
trainer.save_model(cfg['output_dir'])