from llmrec.arguments import DataArguments
from llmrec.utils import load_json

import random
import importlib
from functools import reduce
from typing import Any, Tuple, List, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RecDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        data_repr_encoders = None,
        model_repr_encoders = None,
    ):
        self.dataset = sum([load_json(data_file) for data_file in data_args.rec_training_data_url], [])
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer
        self.data_repr_encoders = [] if data_repr_encoders is None else data_repr_encoders
        self.model_repr_encoders = [] if model_repr_encoders is None else model_repr_encoders
        self.data_args = data_args
        self.data_collator = None

        if data_args.data_collator_class is not None:
            data_collator_core = getattr(importlib.import_module('llmrec.datasets.collators'), data_args.data_collator_class)
            self.data_collator = data_collator_core(
                tokenizer=tokenizer,
                model_repr_max_length=data_args.model_repr_max_length,
                data_repr_max_length=data_args.data_repr_max_length
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item, apply_repr_encode=True) -> Tuple[Any, Any]:
        data_repr = self.dataset[item]['data_repr']
        model_repr = self.dataset[item]['model_repr']

        if apply_repr_encode:
            data_repr = reduce(lambda x, func: func.apply(x), self.data_repr_encoders, data_repr)
            model_repr = reduce(lambda x, func: func.apply(x), self.model_repr_encoders, model_repr)

        return (data_repr, model_repr)
