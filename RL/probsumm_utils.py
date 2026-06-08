import re, pandas
from datasets import Dataset

MODEL_ID = '/home1/shared/Models/Llama-3.2-1B-Instruct'

DRBENCH_TRAIN_PATH = 'DrBench/Csv/summ_0821_train.csv'
DRBENCH_DEV_PATH = 'DrBench/Csv/summ_0821_dev.csv'

# The system prompt instructs the model to reason first, then output a
# semicolon-separated problem list inside <answer> tags so reward functions
# can parse it unambiguously.
SYSTEM_PROMPT = (
    "You are a physician. Based on the clinical note assessment section provided, "
    "identify the most important problems/diagnoses. "
    "First reason through the note in <think> </think> tags, then output the problems "
    "as a semicolon-separated list inside <answer> </answer> tags. "
    "Example: <think> reasoning here </think>"
    "<answer> problem one; problem two; problem three </answer>")


def load_dataset_from_csv(csv_path):
    """Return a HuggingFace Dataset with 'assessment' and 'answer' columns."""

    df = pandas.read_csv(csv_path, dtype='str')
    rows = []

    for assm, summ in zip(df['Assessment'], df['Summary']):
        if isinstance(assm, str) and isinstance(summ, str):
            summ = summ.replace('#', '').replace(':', '').strip()
            rows.append({'assessment': assm, 'answer': summ})

    return Dataset.from_list(rows)


def make_conversation(example):
    """Map dataset row to the prompt/answer format expected by GRPOTrainer."""

    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'### Assessment Section ###\n\n{example["assessment"]}'},
        ],
    }


def extract_answer(generated_text):
    """Return the content of the <answer> tag, or the full text if absent."""

    match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else generated_text.strip()


def normalize(text):
    """Lowercase and collapse whitespace."""

    return re.sub(r'\s+', ' ', text.lower().strip())
