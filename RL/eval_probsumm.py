import torch, os, numpy, sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Lib'))
import data as lib_data

from probsumm_utils import (MODEL_ID, DRBENCH_DEV_PATH,
                             load_dataset_from_csv, make_conversation,
                             extract_answer)

TRAINED_MODEL_DIR = 'Model'


def evaluate(model, tokenizer, dataset):
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
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id)

        input_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

        # Score against only the <answer> portion so <think> tokens don't inflate Rouge-L
        generated_answer = extract_answer(generated_text)
        reference = example['answer']

        f1 = lib_data.calc_rougel(generated_answer.lower(), reference.lower())
        f1s.append(f1)

    return numpy.mean(f1s)


data_base_path = os.environ['DATA_ROOT']
dev_csv_path = os.path.join(data_base_path, DRBENCH_DEV_PATH)

print('Loading dev set...')
dev_dataset = load_dataset_from_csv(dev_csv_path)
dev_dataset = dev_dataset.map(make_conversation)
print(f'Dev examples: {len(dev_dataset)}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)

# --- Base model ---
print(f'\nEvaluating base model: {MODEL_ID}')
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype='auto', device_map='auto')
base_f1 = evaluate(base_model, tokenizer, dev_dataset)
del base_model
torch.cuda.empty_cache()

# --- Trained model ---
print(f'\nEvaluating trained model: {TRAINED_MODEL_DIR}')
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype='auto', device_map='auto')
trained_model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_DIR)
trained_f1 = evaluate(trained_model, tokenizer, dev_dataset)

# --- Results ---
print('\n=== ProbSumm Evaluation Results (Rouge-L F1) ===')
print(f'Base model:    {base_f1:.4f}')
print(f'Trained model: {trained_f1:.4f}')
print(f'Improvement:   {trained_f1 - base_f1:+.4f}')