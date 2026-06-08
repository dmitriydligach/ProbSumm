#!/usr/bin/env python3

import transformers, torch, os, numpy, sys, pandas
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Lib'))
import data

lama_size = '1B'
drbench_dev_path = 'DrBench/Csv/summ_0821_dev.csv'
model_path = f'/home1/shared/Models/Llama-3.2-{lama_size}-Instruct'

def load_data(csv_path):
  """Return (assessment, summary) pairs from CSV"""

  df = pandas.read_csv(csv_path, dtype='str')
  pairs = []

  for assm, summ, subj in zip(df['Assessment'], df['Summary'], df['S']):
    if isinstance(assm, str) and isinstance(summ, str):
      summ = summ.replace('#', '').replace(':', '')
      pairs.append((assm, summ))

  return pairs

def main():

  dev_path = os.path.join(base_path, drbench_dev_path)
  pairs = load_data(dev_path)

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    dtype=torch.bfloat16)
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

    # with messages input + return_full_text=False the output is a list of message dicts
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