import torch, os, numpy, sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Lib'))
import data as lib_data

from probsumm_utils import (load_config, load_dataset_from_csv, make_conversation,
                             extract_answer)

cfg = load_config()


def evaluate(model, tokenizer, dataset, output_file=None):
    """Run greedy decoding on all examples and return average Rouge-L."""

    model.eval()
    f1s = []

    for example in tqdm(dataset, desc='Evaluating'):
        prompt_text = tokenizer.apply_chat_template(
            example['prompt'], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg['generation']['max_new_tokens'],
                do_sample=cfg['generation']['do_sample'],
                pad_token_id=tokenizer.eos_token_id)

        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        # Score against only the <answer> portion so <think> tokens don't inflate Rouge-L
        generated_answer = extract_answer(generated_text)
        reference = example['answer']

        f1 = lib_data.calc_rougel(generated_answer.lower(), reference.lower())
        f1s.append(f1)

        if output_file:
            output_file.write(f'Input: {example["input_text"]}\n')
            output_file.write(f'Reference: {reference}\n')
            output_file.write(f'Generated:\n{generated_text}\n')
            output_file.write(f'Rouge-L: {f1:.4f}\n')
            output_file.write('-' * 80 + '\n')

    return numpy.mean(f1s)


data_base_path = os.environ['DATA_ROOT']
train_csv_path = os.path.join(data_base_path, cfg['data']['train_path'])

print('Loading dev set (last 10% of train)...')
full_dataset = load_dataset_from_csv(train_csv_path)
split = int(len(full_dataset) * 0.9)
dev_dataset = full_dataset.select(range(split, len(full_dataset)))

tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], clean_up_tokenization_spaces=False)

def truncate_prompt(example):
    tokens = tokenizer(example['input_text'], truncation=True, max_length=cfg['training']['max_prompt_length'])
    example['input_text'] = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
    return example

dev_dataset = dev_dataset.map(truncate_prompt)
dev_dataset = dev_dataset.map(make_conversation)
print(f'Dev examples: {len(dev_dataset)}')

# --- Base model ---
print(f'\nEvaluating base model: {cfg["model_id"]}')
base_model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], torch_dtype='auto', device_map='auto')
base_f1 = evaluate(base_model, tokenizer, dev_dataset)
del base_model
torch.cuda.empty_cache()

# --- Trained model ---
print(f'\nEvaluating trained model: {cfg["output_dir"]}')
base_model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], torch_dtype='auto', device_map='auto')
trained_model = PeftModel.from_pretrained(base_model, cfg['output_dir'])
output_path = os.path.join(os.path.dirname(__file__), 'eval_output.txt')
with open(output_path, 'w') as f:
    trained_f1 = evaluate(trained_model, tokenizer, dev_dataset, output_file=f)

# --- Results ---
print('\n=== ProbSumm Evaluation Results (Rouge-L F1) ===')
print(f'Base model:    {base_f1:.4f}')
print(f'Trained model: {trained_f1:.4f}')
print(f'Improvement:   {trained_f1 - base_f1:+.4f}')