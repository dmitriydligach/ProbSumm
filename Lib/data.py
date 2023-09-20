#!/usr/bin/env python3

import os, pandas, string, datasets

drbench_dev_path = 'DrBench/Csv/summ_0821_dev.csv'
drbench_train_path = 'DrBench/Csv/summ_0821_train.csv'

system_prompt = 'You are a physician. Please list as a semicolon separated list ' \
                'the most important problems/diagnoses based on the progress note ' \
                'text below. Only list the problems/diagnoses and nothing else. ' \
                'Be concise.'

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

      input_text = f'### Assessment Section ###\n\n{assm}\n\n' \
                   f'### Problem List ###\n\n{summ}'
      prompt = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>' \
               f'\n\n{input_text} [/INST]\n\n'
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
  data = csv_to_fine_tune_data(train_path)
  print(data['text'][10])

  # dev_path = os.path.join(base_path, drbench_dev_path)
  # input, output = csv_to_zero_shot_data(dev_path)[10]
  # print(input)
  # print('------------')
  # print(output)