"""Use the same explicit text protocol for Base models without native chat templates."""
from verl.utils.dataset.rl_dataset import RLHFDataset
from algescope.training.prompts import BASE_TEMPLATE


class AlgeScopeDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        if config.get('filter_overlong_prompts') or config.get('truncation') != 'error':
            raise ValueError('AlgeScope requires full datasets and explicit overlength errors')
        tokenizer.chat_template = BASE_TEMPLATE
        super().__init__(data_files, tokenizer, config, processor=None)
