import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
import re


class Custom_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            dataset_name,
            valid_subset_path,
            type_path,
            input_length,
            output_length,
            args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path
        self.valid_subset_path = valid_subset_path

        if self.type_path == 'train':
            self.dataset = pd.read_csv(dataset_name, lineterminator='\n')
            batch_size = self.args.train_batch_size * \
                self.args.gradient_accumulation_steps * self.args.ngpu
            if len(self.dataset) != batch_size:
                raise Exception(
                    "Effective batch size should be the same as length of train set")

        else:
            if '.csv' not in self.dataset_name:  # load from huggingface hub
                if valid_subset_path:
                    dataset = load_dataset(
                        self.dataset_name,
                        valid_subset_path,
                        split=type_path,
                        ignore_verifications=True)
                else:
                    dataset = load_dataset(
                        self.dataset_name,
                        split=type_path,
                        ignore_verifications=True)
                self.dataset = dataset.to_pandas()
            else:
                self.dataset = pd.read_csv(dataset_name, lineterminator='\n')

        # About 4 examples have one more or one less class for some reason,
        # they will cause dataloader error
        if self.dataset_name == 'ai2_arc':
            self.dataset['length'] = self.dataset['choices'].apply(
                lambda x: len(x['text']))
            self.dataset = self.dataset[self.dataset['length'] == 4]

    def __len__(self):
        return len(self.dataset)

    def input_to_target(self, input):
        input_s = input.split(' ')
        input_ = " ".join(input_s[:len(input_s) - 1])
        target = " " + input_s[len(input_s) - 1]
        return input_, target

    def convert_to_features(self, example_batch):
        try:
            doc_id = torch.tensor(example_batch['doc_id'], dtype=torch.int)
        except KeyError:
            doc_id = ''

        choices = []
        answer_index = 0
        task, task_type = '', ''
        if self.type_path == 'train':
            input_ = example_batch['text']
            target_ = example_batch['text']
        else:
            if 'lambada' in self.dataset_name:
                input_, target_ = self.input_to_target(example_batch['text'])
                task_type = 'completion'
                task = 'lambada'
            elif self.dataset_name == 'piqa':
                input_ = example_batch['goal']
                choices = [
                    ' ' + example_batch['sol1'],
                    ' ' + example_batch['sol2']]
                target_ = choices[int(example_batch['label'])]
                answer_index = int(example_batch['label'])
                task_type = 'classification'
            elif self.dataset_name == 'hellaswag':
                input_ = example_batch['ctx']
                choices = []
                choices = [' ' + c for c in example_batch['endings']]
                target_ = choices[int(example_batch['label'])]
                answer_index = int(example_batch['label'])
                task_type = 'classification'
            elif self.dataset_name == 'ai2_arc':
                input_ = example_batch['question']
                choices = [' ' + c for c in example_batch['choices']['text']]
                answer_index = example_batch['choices']['label'].tolist().index(
                    example_batch['answerKey'])
                target_ = choices[answer_index]
                task_type = 'classification'
            elif self.dataset_name == 'winogrande':
                input_, rest = example_batch['sentence'].split(' _')
                choices = [
                    ' ' + example_batch['option1'] + rest,
                    ' ' + example_batch['option2'] + rest]
                answer_index = int(
                    example_batch['answer']) - 1  # Label are '1' or '2'
                target_ = choices[answer_index]
                task_type = 'classification'
            elif self.dataset_name == 'math_qa':
                input_ = example_batch['Problem']
                choices = [c[4:].rstrip(" ,") for c in re.findall(
                    r"[abcd] \) .*?, |e \) .*?$", example_batch["options"])]
                answer_index = [
                    'a', 'b', 'c', 'd', 'e'].index(
                    example_batch['correct'])
                target_ = choices[answer_index]
                task_type = 'classification'
            elif 'pubmed_qa' in self.dataset_name:
                input_ = f"Context: {example_batch['abstract']}\nQuestion: {example_batch['question']}\nAnswer:"
                choices = [' yes', ' maybe', ' no']
                answer_index = ['yes', 'maybe', 'no'].index(
                    example_batch['final_decision'])
                target_ = choices[answer_index]
                task = 'pubmed_qa'
                task_type = 'classification'
            elif self.dataset_name == 'super_glue' and self.valid_subset_path == 'copa':
                input_ = example_batch['premise']
                choices = [
                    ' ' + example_batch['choice1'],
                    ' ' + example_batch['choice2']]
                answer_index = int(example_batch['label'])
                target_ = choices[answer_index]
                task_type = 'classification'
            elif 'pile' in self.dataset_name:
                input_, target_ = example_batch['text'], example_batch['text']
                task = 'pile'
                task_type = 'ppl'
            elif 'wikitext' in self.dataset_name:
                input_, target_ = example_batch['text'], example_batch['text']
                task = 'wikitext'
                task_type = 'ppl'
            else:
                input_, target_ = example_batch['text'], example_batch['text']
                task = 'target'
                task_type = 'target'

        if not task:
            if self.valid_subset_path:
                task = f'{self.dataset_name}_{self.valid_subset_path}'
            else:
                task = f'{self.dataset_name}'

        source = self.tokenizer(
            input_,
            max_length=self.input_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt")

        add_target_special_tokens = False if task_type == 'completion' else True
        targets = self.tokenizer(
            target_,
            max_length=self.output_length,
            add_special_tokens=add_target_special_tokens,
            padding='max_length',
            truncation=True,
            return_tensors="pt")
        # targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length,
        # padding='max_length', truncation=True, return_tensors="pt")
        return source, targets, doc_id, task, task_type, choices, answer_index

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        source, targets, doc_id, task, task_type, choices, answer_index = self.convert_to_features(
            data)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "doc_id": doc_id,
                "task": task,
                "task_type": task_type,
                "choices": choices,
                "answer_index": answer_index}
