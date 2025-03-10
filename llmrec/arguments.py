import os
import transformers
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "Path or identifier for a pretrained model from huggingface.co/models."}
    )
    data_and_model_repr_config_url: str = field(
        default='0_meta_info.json',
        metadata={"help": "URL or file path for data & model repr config file (meta_info)."}
    )
    rec_model_class: str = field(
        default='ModelSAT',
        metadata={"help": "Class of recommendation model."}
    )
    data_repr_encoders: Optional[str] = field(
        default=None,
        metadata={"help": "Class of data representation encoders."}
    )
    model_repr_encoders: Optional[str] = field(
        default=None,
        metadata={"help": "Class of LLM (model) representation encoders."}
    )
    peft_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pre-trained PEFT model, or None if not using PEFT."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name or path of a pretrained tokenizer, if different from model_name."}
    )
    use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable LoRA (Low-Rank Adaptation) for fine-tuning."}
    )
    lora_rank: Optional[int] = field(
        default=64,
        metadata={"help": "Rank dimension for LoRA, affecting complexity and cost."}
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA updates."}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for LoRA modules for regularization."}
    )

    def lora_target_modules_default_list() -> List[str]:
        return ["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lora_target_modules_default_list,
        metadata={"help": "Specific model modules targeted by LoRA for adaptation."}
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Modules' parameters to save for LoRA."}
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable flash attention for efficient self-attention."}
    )
    cache_dir: Optional[str] = field(
        default="tmp",
        metadata={"help": "Directory for model cache storage."}
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "Access token for downloading private models from Hugging Face."}
    )

@dataclass
class DataArguments:
    rec_training_data_url: str = field(
        default='capit-1.0.json',
        metadata={"help": "URL or file path for recommendation training data in JSON format."}
    )
    rec_training_dataset_class: str = field(
        default='ContrastiveRecDataset',
        metadata={"help": "Class of the recommendation training data."}
    )
    model_repr_data_url: str = field(
        default='1_model_reprs.json',
        metadata={"help": "URL or file path for model representation file."}
    )
    data_and_model_interaction_url: str = field(
        default='2_interactions.json',
        metadata={"help": "URL or file path for data & model interaction file."}
    )
    data_collator_class: Optional[str] = field(
        default=None,
        metadata={"help": "Class for data collation during training."}
    )
    contrastive_train_group_size: Optional[int] = field(
        default=8,
        metadata={"help": "Number of samples per group for contrastive setting."}
    )
    contrastive_num_sample: Optional[int] = field(
        default=10000,
        metadata={"help": "Number of training data."}
    )
    contrastive_num_pos: Optional[int] = field(
        default=1,
        metadata={"help": "Number of positive samples per group."}
    )
    model_repr_max_length: Optional[int] = field(
        default=256,
        metadata={"help": "Max token length for model's input sequence."}
    )
    data_repr_max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max token length for query representation input."}
    )

    def preprocess(self, data_path=None, suffix_name=''):
        self.rec_training_data_url = self.rec_training_data_url.split(',')
        if data_path:
            self.rec_training_data_url = [os.path.join(data_path, f'{i}{suffix_name}') for i in self.rec_training_data_url]

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    loss_type: str = field(
        default='binary_cross_entropy',
        metadata={"help": "Loss function type for training."}
    )
    rec_trainer_class: str = field(
        default='TransformersTrainer',
        metadata={"help": "Class of trainer."}
    )
