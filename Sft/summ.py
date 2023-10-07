#!/usr/bin/env python3

import transformers, torch, os, numpy, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from peft import PeftModel

sys.path.append('../Lib/')
import data

lama_model = '7b-chat'
model_path = f'/home/dima/Models/Llama/Llama-2-{lama_model}-hf'
print('base model:', model_path)

drbench_dev_path = 'DrBench/Csv/summ_0821_dev.csv'
lora_model = '/home/dima/Git/LLM/Sft/Output/'
print('lora model:', lora_model)

def main():
  """Ask for input and feed into llama2"""

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    load_in_8bit=True)
  model = PeftModel.from_pretrained(model, lora_model)
  pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16, # make sure this matches SFT
    device_map='auto')

  dev_path = os.path.join(base_path, drbench_dev_path)
  inputs_and_outputs = data.csv_to_apaca_zero_shot_data(dev_path)

  f1s = []
  for input_text, reference_output in inputs_and_outputs:

    generated_outputs = pipeline(
      input_text,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      temperature=0.001,
      max_length=512)

    print('-'*50)
    print(generated_outputs[0]['generated_text'])
    print(f'\nreference summary: {reference_output}\n')

    # remove the prompt from output and evaluate
    response = generated_outputs[0]['generated_text'][len(input_text):]
    f1 = data.calc_rougel(response.lower(), reference_output.lower())
    f1s.append(f1)

  print('average f1:', numpy.mean(f1s))

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']
  main()
