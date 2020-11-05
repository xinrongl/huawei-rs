import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split_filename,
        classes=None,
        augmentation=None,
        preprocessing=None,
        mode=None,
    ):
        with open(split_filename) as f:
            _ids = f.readlines()
        self.filenames = list(map(lambda x: x.strip("\n") + ".png", _ids))
        if mode == "test":
            self.filenames = self.filenames[:100]
        self.image_dir = f"{data_dir}/images"
        self.label_dir = f"{data_dir}/labels"
        self.class_values = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        filename = self.filenames[i]
        # read data
        image = cv.imread(f"{self.image_dir}/{filename}")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(f"{self.label_dir}/{filename}", -1)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        return image, mask

    def __len__(self):
        return len(self.filenames)
