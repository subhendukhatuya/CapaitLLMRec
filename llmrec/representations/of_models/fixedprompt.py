class ModelReprFixedPrompt:
    prompt = 'Model description: {model_repr}'

    def __init__(self, prompt=None, **kwargs):
        if prompt is not None:
            ModelReprFixedPrompt.prompt = prompt
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def apply(item):
        return ModelReprFixedPrompt.prompt.format(model_repr=item)