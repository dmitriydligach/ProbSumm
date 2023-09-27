#!/usr/bin/env python3

import os, pandas, string, datasets, json
from rouge_score import rouge_scorer

drbench_train_path = 'DrBench/Csv/summ_0821_train.csv'
drbench_dev_path = 'DrBench/Csv/summ_0821_dev.csv'

system_prompt = 'You are a physician. Please list as a semicolon separated list ' \
                'the most important problems/diagnoses based on the progress note ' \
                'text below. Only list the problems/diagnoses and nothing else. ' \
                'Be concise.'

def calc_rougel(generated_text, reference_text):
  """Compute Rouge-L score"""

  # {'rougeL': Score(precision=0.5, recall=0.6, fmeasure=0.5)}
  scorer = rouge_scorer.RougeScorer(['rougeL'])
  scores = scorer.score(reference_text, generated_text)
  f1 = scores['rougeL'].fmeasure

  return f1

def csv_to_json(input_csv_path, output_json_path):
  """Convert to json to use for SFT"""

  df = pandas.read_csv(input_csv_path, dtype='str')

  # list of dictionaries to save as json
  samples = []

  for assm, summ, in zip(df['Assessment'], df['Summary']):

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(summ) == str:
      assm = ''.join(c for c in assm if c in string.printable)
      summ = ''.join(c for c in summ if c in string.printable)
      summ = summ.replace('#', '') # cleanup
      summ = summ.replace(':', '') # cleanup

      sample = {'instruction': system_prompt,
                'input': assm,
                'output': summ}
      samples.append(sample)

  json_file = open(output_json_path, 'w')
  json.dump(samples, json_file, indent=2)

def csv_to_fine_tune_data(data_csv_path):
  """Format training data for fine-tuning and make a HF dataset"""

  df = pandas.read_csv(data_csv_path, dtype='str')

  # input/output pairs
  train_samples = []

  for assm, summ, _ in zip(df['Assessment'], df['Summary'], df['S']):

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(summ) == str:
      assm = ''.join(c for c in assm if c in string.printable)
      summ = ''.join(c for c in summ if c in string.printable)
      summ = summ.replace('#', '') # cleanup
      summ = summ.replace(':', '') # cleanup

      input_text = f'### Assessment Section ###\n\n{assm}'
                   # f'### Problem List ###\n\n{summ}'
      prompt = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>' \
               f'\n\n{input_text} [/INST]\n\n' \
               f'### Problem List ###\n\n{summ}'
      train_samples.append(prompt)

  data = datasets.Dataset.from_dict({'text': train_samples})
  # split_data = data.train_test_split(test_size=0.2, shuffle=True)
  # return split_data

  return data

def csv_to_zero_shot_data(data_csv_path, include_subjective=False):
  """Get summarization input/output pair tuples"""

  df = pandas.read_csv(data_csv_path, dtype='str')

  # input/output pairs
  ios = []

  for assm, summ, subj in zip(df['Assessment'], df['Summary'], df['S']):

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(summ) == str and type(subj) == str:
      assm = ''.join(c for c in assm if c in string.printable)
      summ = ''.join(c for c in summ if c in string.printable)
      subj = ''.join(c for c in subj if c in string.printable)
      summ = summ.replace('#', '') # cleanup
      summ = summ.replace(':', '') # cleanup

      if include_subjective:
        input_text = f'### Subjective Section ###\n\n{subj}\n\n' \
                     f'### Assessment Section ###\n\n{assm}'
      else:
        input_text = f'### Assessment Section ###\n\n{assm}'

      prompt = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>' \
               f'\n\n{input_text} [/INST]\n\n'

      ios.append((prompt, summ))

  return ios

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']

  train_path = os.path.join(base_path, drbench_train_path)
  dev_path = os.path.join(base_path, drbench_dev_path)

  csv_to_json(train_path, '/home/dima/Temp/summ_train.json')
  csv_to_json(dev_path, '/home/dima/Temp/summ_dev.json')

  # train_path = os.path.join(base_path, drbench_train_path)
  # data = csv_to_fine_tune_data(train_path)
  # print(data['text'][10])

  # dev_path = os.path.join(base_path, drbench_dev_path)
  # input, output = csv_to_zero_shot_data(dev_path)[10]
  # print(input)
  # print('------------')
  # print(output)