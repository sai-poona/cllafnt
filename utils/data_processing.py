import json
from pathlib import Path
from datasets import load_dataset
from typing import Dict
import logging

from llama_recipes.data.concatenator import Concatenator

logging.basicConfig(level=logging.INFO)


def find_data_files(path: str, file_extension: str):
    allowed_extensions = [
        'jsonl'
    ]
    if file_extension not in allowed_extensions:
        raise ValueError(
            f'Provided file extension "{file_extension}" is not in the list of '
            'allowed extensions. file_extension must be one of \n'
            f'{json.dumps(allowed_extensions, indent=4)}'
        )
    return sorted([str(x) for x in Path(path).glob(f'*.{file_extension}')])


def load_train_and_validation_data(args):
    data_files: Dict[str, str] = {}

    if 'train_dir' in args:
        data_files['train'] = find_data_files(path=args['train_dir'], file_extension=args['file_extension'])
    if 'validation_dir' in args:
        data_files['validation'] = find_data_files(path=args['validation_dir'], file_extension=args['file_extension'])

    if args['file_extension'] == 'jsonl':
        extension = 'json'

    datasets = load_dataset(extension, data_files=data_files)
    return datasets


def get_template(filename, end_key):
    template_file = Path(filename)
    if template_file.exists():
        with open(template_file, "r") as f:
            template = json.load(f)
        if 'prompt' not in template:
            raise ValueError('No prompt found in the template')
        if 'completion' not in template:
            raise ValueError('No completion found in the template')

        if end_key is None:
            end_key = "### End"

        prompt_template = f"""### Instruction:

{template['prompt']}

### Response:

{template['completion']}

{end_key}"""
    else:
        raise ValueError(f'Unable to find template file {filename}')
    return prompt_template


def preprocess_dataset(args, tokenizer):
    loaded_dataset = load_train_and_validation_data(args)
    prompt = get_template(args['prompt_template'], tokenizer.eos_token)

    def process_dataset(dataset):
        def apply_prompt_template(sample):
            return {'text': prompt.format(**sample)}

        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

        dataset = dataset.map(
            lambda sample: tokenizer(sample['text']),
            batched=True,
            remove_columns=list(dataset.features),
        )

        if args['max_input_length'] == -1:
            args['max_input_length'] = 2048
            logging.info('Using the default value of max_input_length=2048.')
        dataset = dataset.map(Concatenator(chunk_size=args['max_input_length']), batched=True)
        return dataset

    if 'validation' in loaded_dataset:
        dataset_train = process_dataset(loaded_dataset['train'])
        dataset_val = process_dataset(loaded_dataset['validation'])
    else:
        full_dataset = process_dataset(loaded_dataset['train'])
        dataset = full_dataset.train_test_split(test_size=args['validation_split_ratio'], seed=args['seed'])
        dataset_train, dataset_val = dataset['train'], dataset['test']
    return dataset_train, dataset_val