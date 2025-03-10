class DataReprFixedPrompt:
    prompt = 'Test Sample: {data_repr}'
    delete_sub_strs = [
        '\nAnswer:'
    ]

    def __init__(self, prompt=None, **kwargs):
        if prompt is not None:
            DataReprFixedPrompt.prompt = prompt
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def apply(item):
        for i_del_str in DataReprFixedPrompt.delete_sub_strs:
            item = item.replace(i_del_str, '')
        return DataReprFixedPrompt.prompt.format(data_repr=item)