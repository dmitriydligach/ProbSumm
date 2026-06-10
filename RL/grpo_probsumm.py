import re, os
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from probsumm_utils import (MODEL_ID, DRBENCH_TRAIN_PATH,
                             load_dataset_from_csv, make_conversation,
                             extract_answer, normalize)

OUTPUT_DIR = 'Model'

data_base_path = os.environ['DATA_ROOT']
train_csv_path = os.path.join(data_base_path, DRBENCH_TRAIN_PATH)

print('Loading training data...')
train_dataset = load_dataset_from_csv(train_csv_path)
train_dataset = train_dataset.map(make_conversation)

print(f'Training examples: {len(train_dataset)}')

local_rank = int(os.environ.get('LOCAL_RANK', -1))
device_map = 'auto' if local_rank == -1 else {'': local_rank}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)

print(f'Loading model: {MODEL_ID}...')
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype='auto', device_map=device_map)

lora_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'])

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
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    remove_unused_columns=False,  # keep 'answer' column for problem_coverage_reward
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    bf16=True,
    max_completion_length=768,
    num_generations=8,
    report_to=['tensorboard'],
    logging_steps=10,
    push_to_hub=False,
    save_strategy='steps',
    save_steps=50)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, problem_coverage_reward],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer)

print('Starting GRPO training...')
trainer.train()

print(f'Saving model to {OUTPUT_DIR}...')
trainer.save_model(OUTPUT_DIR)