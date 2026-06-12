#!/usr/bin/env python3

import transformers, torch, os, numpy, sys, pandas
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Lib'))
import data

lama_size = '1B'
base_model_path = f'/home1/shared/Models/Llama-3.2-{lama_size}-Instruct'
adapter_path = 'Output'

train_csv_path = 'ProbSumm/BioNLP2023-1A-Train.csv'

def load_data(csv_path):
  """Return (assessment, summary) pairs from CSV"""

  df = pandas.read_csv(csv_path, dtype='str')
  pairs = []

  for assm, summ in zip(df['Assessment'], df['Summary']):
    if isinstance(assm, str) and isinstance(summ, str):
      summ = summ.replace('#', '').replace(':', '')
      pairs.append((assm, summ))

  return pairs

def main():

  csv_path = os.path.join(base_path, train_csv_path)
  all_pairs = load_data(csv_path)
  split = int(len(all_pairs) * 0.9)
  pairs = all_pairs[split:]

  tokenizer = AutoTokenizer.from_pretrained(base_model_path)
  tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map='auto',
    dtype=torch.bfloat16)
  model = PeftModel.from_pretrained(model, adapter_path)
  model.eval()

  pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer)

  f1s = []
  inference_times = []

  for assessment, reference_output in pairs:

    messages = [
      {'role': 'system', 'content': data.system_prompt},
      {'role': 'user', 'content': f'### Assessment Section ###\n\n{assessment}'},
    ]

    start = time()
    generated_outputs = pipeline(
      messages,
      do_sample=False,
      num_return_sequences=1,
      max_new_tokens=256,
      return_full_text=False)
    end = time()
    inference_times.append(end - start)

    result = generated_outputs[0]['generated_text']
    generated_text = result[-1]['content'] if isinstance(result, list) else result

    print('************************************************\n')
    print(generated_text)
    print(f'\n### Reference Summary ###\n\n{reference_output}\n')

    f1 = data.calc_rougel(generated_text.lower(), reference_output.lower())
    f1s.append(f1)

  av_inf_time = numpy.mean(inference_times)
  print(f'\naverage inference time: {av_inf_time} seconds')
  print('average f1:', numpy.mean(f1s))

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']
  main()
