import albumentations as albu


def get_training_augmentation():
    train_transform = [
        # albu.RandomRotate90(p=1),
        # albu.HorizontalFlip(p=0.25)
        # albu.RGBShift(
        #     r_shift_limit=20,
        #     g_shift_limit=20,
        #     b_shift_limit=20,
        #     always_apply=False,
        #     p=0.5,
        # ),
        # albu.CLAHE(p=0.5)
        # albu.IAAAdditiveGaussianNoise(p=1),
        # albu.IAAPerspective(p=0.5),
        # albu.OneOf(
        #     [albu.RandomBrightness(p=1), albu.RandomGamma(p=1)],
        #     p=0.9,
        # ),
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        # albu.OneOf(
        #     [albu.RandomContrast(p=1), albu.HueSaturationValue(p=1)],
        #     p=0.9,
        # ),
        # albu.Normalize(
        #     mean=(0.355, 0.384, 0.359),
        #     std=(0.207, 0.202, 0.21),
        # ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        # albu.Normalize(
        #     mean=(0.355, 0.384, 0.359),
        #     std=(0.207, 0.202, 0.21),
        # )
    ]
    return albu.Compose(test_transform)


def get_test_augmentation():
    test_transform = [
        # albu.Normalize(
        #     mean=(0.355, 0.384, 0.359),
        #     std=(0.207, 0.202, 0.21),
        # )
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    if len(x.shape) > 2:
        return x.transpose(2, 0, 1).astype("float32")
    else:
        return x.astype("int64")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]

    return albu.Compose(_transform)