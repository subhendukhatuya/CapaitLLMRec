from llmrec.arguments import ModelArguments, TrainingArguments
from llmrec.trainers.utils import safe_save_model_for_hf_trainer

import os
import pathlib
from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import Trainer


class GenTrainer(Trainer):
    def __init__(self, *args, use_lora: bool = False, **kwargs):
        super(GenTrainer, self).__init__(*args, **kwargs)
        self.use_lora = use_lora

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if not self.use_lora:
            super()._save(output_dir, state_dict)
            return

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

class TransformersTrainer:
    def __init__(
        self,
        train_dataset: Dataset,
        model: nn.Module,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ):
        self.trainer = GenTrainer(
            train_dataset=train_dataset,
            model=model,
            tokenizer=model.tokenizer,
            args=training_args,
            data_collator=train_dataset.data_collator,
            use_lora=model_args.use_lora
        )
        self.model = model
        self.training_args = training_args

    def train(self):
        if list(pathlib.Path(self.training_args.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()

    def save(self):
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer,
            output_dir=self.training_args.output_dir
        )
