import re, json, pandas
from pathlib import Path
from datasets import Dataset
from rouge_score import rouge_scorer as rouge_scorer_lib

_scorer = rouge_scorer_lib.RougeScorer(['rougeL'])

_CONFIG_PATH = Path(__file__).parent / 'config.json'


def load_config():
    """Load and return the JSON config from the RL directory."""
    with open(_CONFIG_PATH) as f:
        return json.load(f)

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
    """Return a HuggingFace Dataset with 'assessment', 'subjective', and 'answer' columns."""

    df = pandas.read_csv(csv_path, dtype='str')
    rows = []

    for assm, summ, subj in zip(df['Assessment'], df['Summary'], df['Subjective Sections']):
        if isinstance(assm, str) and isinstance(summ, str):
            summ = summ.replace('#', '').replace(':', '').strip()
            rows.append({
                'assessment': assm,
                'subjective': subj if isinstance(subj, str) else '',
                'answer': summ,
            })

    return Dataset.from_list(rows)


def make_conversation(example):
    """Map dataset row to the prompt/answer format expected by GRPOTrainer."""

    user_content = (
        f'### Subjective Section ###\n\n{example["subjective"]}\n\n'
        f'### Assessment Section ###\n\n{example["assessment"]}'
    ) if example['subjective'] else f'### Assessment Section ###\n\n{example["assessment"]}'

    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ],
    }


def extract_answer(generated_text):
    """Return the content of the <answer> tag, or the full text if absent."""

    match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else generated_text.strip()


def normalize(text):
    """Lowercase and collapse whitespace."""

    return re.sub(r'\s+', ' ', text.lower().strip())


def calc_rougel(generated, reference):
    """Return Rouge-L F1 between two strings."""

    return _scorer.score(reference, generated)['rougeL'].fmeasure
