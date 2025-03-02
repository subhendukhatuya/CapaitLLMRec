class DataReprFixedPrompt:
    prompt = 'Test Sample: {data_repr}'

    def __init__(self, prompt=None, **kwargs):
        if prompt is not None:
            DataReprFixedPrompt.prompt = prompt
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def apply(item):
        return DataReprFixedPrompt.prompt.format(data_repr=item)