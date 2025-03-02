import importlib
import torch.nn as nn
from transformers.utils.quantization_config import QuantizationMethod
from llmrec.arguments import ModelArguments, TrainingArguments

class BaseModel(nn.Module):
    def __init__(self,
                 model_args: ModelArguments,
                 training_args: TrainingArguments
                 ):
        super(BaseModel, self).__init__()
        self.model_args = model_args
        self.training_args = training_args
        self._init_data_repr_encoders()
        self._init_model_repr_encoders()

    def _init_data_repr_encoders(self):
        self.data_repr_encoders = []
        if self.model_args.data_repr_encoders is not None:
            self.data_repr_encoders = [
                getattr(
                    importlib.import_module('llmrec.representations'),
                    data_repr_encoder_class
                ) for data_repr_encoder_class in self.model_args.data_repr_encoders.split(',')
            ]

    def _init_model_repr_encoders(self):
        self.model_repr_encoders = []
        if self.model_args.model_repr_encoders is not None:
            self.model_repr_encoders = [
                getattr(
                    importlib.import_module('llmrec.representations'),
                    model_repr_encoder_class
                ) for model_repr_encoder_class in self.model_args.model_repr_encoders.split(',')
            ]

    def to(self, device):
        if hasattr(self, 'model') and not getattr(self.model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            try:
                self.model.to(device)
            except RuntimeError as e:
                pass
        return self

    def eval(self):
        if hasattr(self, 'model'):
            self.model.eval()
        return self

    def tie_weights(self):
        if hasattr(self, 'model') and hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
        return self

    def get_tokenizer(self):
        if hasattr(self, 'tokenizer'):
            return self.tokenizer
        return None
