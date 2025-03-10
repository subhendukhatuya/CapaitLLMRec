from llmrec.arguments import DataArguments
from llmrec.datasets import RecDataset

from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Optional
import random
import math

from functools import reduce


class ContrastiveRecDataset(RecDataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        data_repr_encoders = None,
        model_repr_encoders = None
    ):
        super(ContrastiveRecDataset, self).__init__(data_args, tokenizer, data_repr_encoders, model_repr_encoders)
        sep = "\n"
        self.sep_inputs = tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']
        self.judge_words = 'Yes'
        self.max_length = data_args.model_repr_max_length + data_args.data_repr_max_length

    def __getitem__(self, item) -> List[BatchEncoding]:
        data_repr, model_repr = super().__getitem__(item, apply_repr_encode=False)
        model_repr = reduce(lambda x, cls_core: cls_core.apply(x), self.model_repr_encoders, model_repr)

        data_repr_list = []
        data_repr_list.append(random.choice(data_repr['pos']))
        if len(data_repr['neg']) < self.data_args.contrastive_train_group_size - 1:
            num = math.ceil((self.data_args.contrastive_train_group_size - 1) / len(data_repr['neg']))
            negs = random.sample(data_repr['neg'] * num, self.data_args.contrastive_train_group_size - 1)
        else:
            negs = random.sample(data_repr['neg'], self.data_args.contrastive_train_group_size - 1)
        data_repr_list.extend(negs)
        data_repr_list = [reduce(lambda x, cls_core: cls_core.apply(x), self.data_repr_encoders, i_data_repr) for i_data_repr in data_repr_list]

        query = "Based on the model description and the test sample, predict whether the model can handle test sample by indicating 'Yes' or 'No'."

        model_repr_inputs = self.tokenizer(
            model_repr,
            return_tensors=None,
            max_length=self.data_args.model_repr_max_length + self.data_args.data_repr_max_length // 4,
            truncation=True,
            add_special_tokens=False
        )
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      add_special_tokens=False)['input_ids'] + \
                       self.tokenizer(self.judge_words,
                                      return_tensors=None,
                                      add_special_tokens=False)['input_ids']

        max_length = self.max_length - len(query_inputs) - len(self.sep_inputs)
        inputs = []
        for data_item in data_repr_list:
            data_repr_inputs = self.tokenizer(
                data_item,
                return_tensors=None,
                max_length=self.data_args.data_repr_max_length + self.data_args.model_repr_max_length // 2,
                truncation=True,
                add_special_tokens=False
            )

            bos_inputs_list = []
            if self.tokenizer.bos_token_id is not None and self.tokenizer.bos_token_id != self.tokenizer.pad_token_id:
                bos_inputs_list = [self.tokenizer.bos_token_id]
            item = self.tokenizer.prepare_for_model(
                bos_inputs_list + model_repr_inputs['input_ids'],
                self.sep_inputs + data_repr_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )

            data_repr_inputs['input_ids'] = item['input_ids'] + self.sep_inputs + query_inputs
            data_repr_inputs['attention_mask'] = [1] * len(data_repr_inputs['input_ids'])
            data_repr_inputs['labels'] = data_repr_inputs['input_ids'].copy()
            data_repr_inputs['labels'] = [-100] * (len(data_repr_inputs['input_ids']) - 1) + data_repr_inputs['labels'][(len(data_repr_inputs['input_ids']) - 1):]
            data_repr_inputs.pop('token_type_ids') if 'token_type_ids' in data_repr_inputs.keys() else None
            if 'position_ids' in data_repr_inputs.keys():
                data_repr_inputs['position_ids'] = list(range(len(data_repr_inputs['input_ids'])))
            inputs.append(data_repr_inputs)

        return inputs
