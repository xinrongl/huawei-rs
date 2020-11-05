import random
from pathlib import Path, PurePath

DATA_DIR = Path("/home/xinrong/huawei-rs/data/1024_1024")
TRAIN_SIZE = 0.9


def split(data_dir, train_size):
    if not isinstance(data_dir, PurePath):
        data_dir = Path(data_dir)
    image_id = sorted(data_dir.joinpath("images").glob("*"))
    target_id = sorted(data_dir.joinpath("labels").glob("*"))
    assert len(image_id) == len(
        target_id
    ), "number of images not equal to number of target"

    random.shuffle(image_id)
    _ids = map(lambda x: x.name.split(".")[0], image_id)
    train_ids, val_ids = [], []
    for _id in _ids:
        if random.random() <= train_size:
            train_ids.append(_id)
        else:
            val_ids.append(_id)
    return train_ids, val_ids


def main():
    f1 = open(DATA_DIR.joinpath("train.txt"), "w")
    f2 = open(DATA_DIR.joinpath("val.txt"), "w")

    train_ids, val_ids = split(data_dir=DATA_DIR, train_size=TRAIN_SIZE)
    for train_id in train_ids:
        f1.write(train_id + "\n")
    for val_id in val_ids:
        f2.write(val_id + "\n")
    f1.close()
    f2.close()


if __name__ == "__main__":
    main()
