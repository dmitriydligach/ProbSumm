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
    """Return a HuggingFace Dataset with 'input_text' and 'answer' columns.

    Which CSV columns are included in input_text is controlled by
    data.input_columns in config.json — remove or reorder entries there
    to change what the model sees.
    """

    cfg = load_config()
    input_columns = cfg['data']['input_columns']

    df = pandas.read_csv(csv_path, dtype='str')
    rows = []

    for _, row in df.iterrows():
        summ = row.get('Summary', '')
        if not isinstance(summ, str):
            continue
        summ = summ.replace('#', '').replace(':', '').strip()

        parts = []
        for col_spec in input_columns:
            val = row.get(col_spec['column'], '')
            if isinstance(val, str) and val.strip():
                parts.append(f'### {col_spec["header"]} ###\n\n{val}')

        if parts:
            rows.append({'input_text': '\n\n'.join(parts), 'answer': summ})

    return Dataset.from_list(rows)


def make_conversation(example):
    """Map dataset row to the prompt/answer format expected by GRPOTrainer."""

    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['input_text']},
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
