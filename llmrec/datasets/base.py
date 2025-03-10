from llmrec.arguments import DataArguments
from llmrec.utils import load_json
from llmrec.configs import DATA_PATH

import random
import importlib
from functools import reduce
from typing import Any, Tuple, List, Optional
from collections import defaultdict
import os.path

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
        # self._init_pos_neg_samples(
        #     num_sample=data_args.contrastive_num_sample,
        #     num_pos=data_args.contrastive_num_pos,
        #     num_neg=data_args.contrastive_train_group_size - data_args.contrastive_num_pos
        # )
        random.shuffle(self.dataset)

        self.data_repr_encoders = [] if data_repr_encoders is None else data_repr_encoders
        self.model_repr_encoders = [] if model_repr_encoders is None else model_repr_encoders


        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_collator = None

        if data_args.data_collator_class is not None:
            data_collator_core = getattr(importlib.import_module('llmrec.datasets.collators'), data_args.data_collator_class)
            self.data_collator = data_collator_core(
                tokenizer=tokenizer,
                model_repr_max_length=data_args.model_repr_max_length,
                data_repr_max_length=data_args.data_repr_max_length
            )

    def _init_pos_neg_samples(self, num_sample, num_pos, num_neg):
        model_reprs = load_json(data_args.model_repr_data_url)
        interactions = load_json(data_args.data_and_model_interaction_url)
        prepared_datasets = set([i["dataset_name"] for i in interactions])
        dataset_name2id2data = {
            dataset_name: {
                i['id']: i for i in load_json(os.path.join(DATA_PATH, f"{dataset_name}.json")) \
            } for dataset_name in prepared_datasets
        }

        model_id2model_reprs = {i["model_repr_id"]: i for i in model_reprs}
        model_id2interactions = defaultdict(list)
        for i_interaction in interactions:
            model_id2interactions[i_interaction["model_repr_id"]].append(i_interaction)

        model2pos, model2neg = defaultdict(list), defaultdict(list)
        for model_repr_id in model_id2model_reprs.keys():
            if model_repr_id in model_id2interactions.keys():
                for i_interact in model_id2interactions[model_repr_id]:
                    id2data = dataset_name2id2data[i_interact['dataset_name']]
                    model2pos[model_repr_id].extend([id2data[idx] for idx in i_interact['correct_query_indices']])
                    model2neg[model_repr_id].extend([id2data[idx] for idx in i_interact['incorrect_query_indices']])

        self.dataset = []
        for _ in range(num_sample):
            model_repr_id = random.choice(model_id2model_reprs.keys())
            self.dataset.append(
                {
                    'model_repr': model_id2model_reprs[model_repr_id],
                    'data_repr': {
                        'pos': random.choices(model2pos[model_repr_id], k=min(len(model2pos[model_repr_id]), num_pos)),
                        'neg': random.choices(model2neg[model_repr_id], k=min(len(model2neg[model_repr_id]), num_neg))
                    }
                }
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
