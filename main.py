import importlib

from transformers import HfArgumentParser, set_seed

from llmrec.arguments import ModelArguments, DataArguments, TrainingArguments
from llmrec.configs import DATA_PATH

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.preprocess(DATA_PATH, suffix_name='.json')

    model_core = getattr(importlib.import_module('llmrec.models'), model_args.rec_model_class)
    model = model_core(
        model_args=model_args,
        training_args=training_args
    )
    set_seed(training_args.seed)

    dataset_core = getattr(importlib.import_module('llmrec.datasets'), data_args.rec_training_dataset_class)
    train_dataset = dataset_core(
        data_args=data_args,
        tokenizer=model.tokenizer,
        data_repr_encoders=model.data_repr_encoders,
        model_repr_encoders=model.model_repr_encoders
    )

    trainer_core = getattr(importlib.import_module('llmrec.trainers'), training_args.rec_trainer_class)
    trainer_wrapper = trainer_core(
        train_dataset=train_dataset,
        model=model,
        model_args=model_args,
        training_args=training_args
    )
    trainer_wrapper.train()
    trainer_wrapper.save()

if __name__ == "__main__":
    main()