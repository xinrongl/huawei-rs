# (1) train segmentation model from the scratch using deeplabv3plus and efficientnet-b3 as encoder.
# cd naic
# python train.py --encoder resnext50_32x4d -w imagenet --arch unet -b 4 -lr 5e-5 -wd 5e-6 --num_workers 12 --num_epoch 100 --parallel
import argparse
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn
import yaml
from torch import optim
from torch.utils.data import DataLoader

from src import aug
from src.dataset import CustomDataset
from src.metrics import mIoU
from src.optimizer import RAdam
from src.utils import MyLogger

TIMESTAMP = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

with open("cfgs/cfgs.yaml", "r") as f:
    cfgs = yaml.load(f, yaml.FullLoader)

parser = argparse.ArgumentParser("Train segmentation model with SMP api.")
parser.add_argument("--encoder", default="efficientnet-b5")
parser.add_argument(
    "-w", "--weight", default=None, help="Encoder pretrained weight", required=True
)
parser.add_argument("--activation", default="sigmoid")
parser.add_argument(
    "--arch",
    required=True,
    help="model arch: "
    + " | ".join(
        ["unet", "linkednet", "fpn", "pspnet", "deeplabv3", "deeplabv3plus", "pan"]
    ),
    type=lambda arch: arch.lower(),
)
parser.add_argument("-depth", "--encoder_depth", type=int, default=5)
parser.add_argument("-b", "--batch_size", type=int, default=4)
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument(
    "--save_threshold",
    type=float,
    default=0.5,
    help="Save model if model score greater than threshold",
)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--num_epoch", default=10, type=int)
parser.add_argument("--loglevel", default="INFO")
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--load",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--parallel", action="store_true", help="Use multi-gpus for training"
)
parser.add_argument("--test", action="store_true", help="Test code use small dataset")
args, _ = parser.parse_known_args()

aux_params_dict = dict(pooling="avg", dropout=0.5, activation="softmax", classes=2)

arch_dict = {
    "unet": smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
        decoder_attention_type="scse",
        decoder_use_batchnorm=True,
        aux_params=aux_params_dict,
    ),
    "linknet": smp.Linknet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
    ),
    "fpn": smp.FPN(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
    ),
    "pspnet": smp.PSPNet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
    ),
    "deeplabv3": smp.DeepLabV3(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
    ),
    "deeplabv3plus": smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
        aux_params=aux_params_dict,
    ),
    "pan": smp.PAN(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=2,
        activation=args.activation,
        aux_params=aux_params_dict,
    ),
}


def save_checkpoint(state, filename):
    torch.save(state, filename)


def save_best_checkpoint(max_score, checkpoint_path):
    best_score_suffix = f"{max_score:.4f}.pth"
    pth_files = checkpoint_path.glob("*.pth")
    for pth_file in pth_files:
        if pth_file.name.endswith(best_score_suffix):
            shutil.copy(pth_file, checkpoint_path.join("model_best.pth"))
            break


def main():
    logger.info(f"Loading images from {cfgs['data_dir']}")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.weight)
    train_dataset = CustomDataset(
        data_dir=cfgs["data_dir"],
        split_filename=cfgs["split_filename_train"],
        classes=[0, 1],
        augmentation=aug.get_training_augmentation(),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    valid_dataset = CustomDataset(
        data_dir=cfgs["data_dir"],
        split_filename=cfgs["split_filename_val"],
        classes=[0, 1],
        augmentation=aug.get_validation_augmentation(),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.resume:
        checkpoints = torch.load(args.resume)
        state_dict = OrderedDict()
        for key, value in checkpoints["state_dict"].items():
            tmp = key[7:]
            state_dict[tmp] = value

        model = arch_dict[checkpoints["arch"]]
        model.load_state_dict(state_dict)
        logger.info(
            f"=> loaded checkpoint '{args.resume}' (epoch {args.resume.split('_')[-2]})"
        )

    else:
        model = arch_dict[args.arch]
    optimizer = RAdam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.parallel:
        model = torch.nn.DataParallel(model).cuda()

    metrics = [
        mIoU(threshold=args.threshold),
    ]

    # loss = smp.utils.losses.CrossEntropyLoss()
    # loss = smp.utils.losses.BCELoss()
    loss = smp.utils.losses.JaccardLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience, verbose=True
    )

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    checkpoint_path = Path(
        f"{cfgs['checkpoint_dir']}/{args.arch}_{args.encoder}/{TIMESTAMP:%Y%m%d%H%M}"
    )
    if not args.test:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    max_score = checkpoints["best_iou"] if args.resume else 0.0
    start_epoch = checkpoints["epoch"] + 1 if args.resume else 0
    end_epoch = start_epoch + args.num_epoch
    logger.info(f"Current best score: {max_score:.4f}")
    # optimizer.param_groups[0]["lr"] = args.learning_rate
    logger.info(f"Start learning rate: {optimizer.param_groups[0]['lr']:f}")
    for epoch in range(start_epoch, end_epoch):
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_loss, train_iou = train_logs.values()
        val_loss, val_iou = valid_logs.values()
        logger.info(
            f"epoch [{epoch:03d} | {end_epoch:03d}] | lr: {optimizer.param_groups[0]['lr']:f} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | train iou: {train_iou:.4f} | val iou: {val_iou:.4f}"
        )
        scheduler.step(val_loss)
        if all([max_score < val_iou, val_iou > args.save_threshold, not args.test]):
            max_score = val_iou
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "encoder": args.encoder,
                    "encoder_weight": args.weight,
                    "state_dict": model.state_dict(),
                    "best_iou": max_score,
                    "optimizer": optimizer.state_dict(),
                    "activation": args.activation,
                },
                checkpoint_path.joinpath(f"epoch_{epoch}_{max_score:.4f}.pth"),
            )
            logger.info(f"Save checkpoint at {epoch}.")

    save_best_checkpoint(max_score, checkpoint_path)


if __name__ == "__main__":
    logger_dir = Path(f"{cfgs['log_dir']}/{args.arch}_{args.encoder}")
    logger_dir.mkdir(parents=True, exist_ok=True)
    logger = MyLogger(args.loglevel)
    logger.set_stream_handler()
    if not args.test:
        logger.set_file_handler(f"{logger_dir}/{TIMESTAMP:%Y%m%d%H%M}.log")
    for arg, val in sorted(vars(args).items()):
        logger.info(f"{arg}: {val}")
    logger.info("\n")
    main()
