from data.base_dataset import BaseDataset


class NullDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.opt.max_dataset_size
