from torch.utils.data import Dataset
import numpy as np

class CombineDataset(Dataset):
    def __init__(self,
                 datasets_cfgs,
                 ):
        super().__init__()

        self.datasets = []
        self.datasets_length = []

        self.tokenizer = datasets_cfgs[0].tokenizer
        tokenizer_type = self.tokenizer['type']
        del self.tokenizer['type']
        self.tokenizer = tokenizer_type(**self.tokenizer)

        self._add_special_tokens()

        for i in range(len(datasets_cfgs)):
            datasets_cfgs[i].tokenizer = self.tokenizer

        for dataset_cfg in datasets_cfgs:
            dataset = dataset_cfg['type']
            del dataset_cfg['type']
            dataset = dataset(**dataset_cfg)
            self.datasets.append(dataset)
            self.datasets_length.append(len(dataset))

        self.dataset_threthold = []
        for i, length in enumerate(self.datasets_length):
            if i == 0:
                self.dataset_threthold.append(length)
            else:
                self.dataset_threthold.append(length + self.dataset_threthold[i - 1])

        np.random.seed(42)
        self.shuffled_index = np.arange(self.dataset_threthold[-1])
        np.random.shuffle(self.shuffled_index)

    @property
    def modality_length(self):
        length_list = []
        for dataset in self.datasets:
            for data_dict in dataset.text_data:
                cur_len = len(data_dict['input_ids'])
                if data_dict.get('image', None) is None:
                    cur_len = -cur_len
                length_list.append(cur_len)
        return length_list

    def __len__(self):
        return self.dataset_threthold[-1]

    def __getitem__(self, index):
        index = int(self.shuffled_index[index])
        for i, thred in enumerate(self.dataset_threthold):
            if index < thred:
                break


        if i == 0:
            _index = index
        else:
            _index = index - self.dataset_threthold[i - 1]

        return self.datasets[i][_index]

    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        # add for visual prompt
        region_tokens = ['<region>']
        point_tokens = ['<mark>']
        special_tokens = segmentation_tokens + phrase_tokens + region_tokens + point_tokens

        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        return