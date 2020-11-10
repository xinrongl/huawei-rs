import sys
import matplotlib.pyplot as plt

logfile = sys.argv[1].split()[-1]
logfilename = logfile.split("/")[-1].strip(".log")


def get_val(vals):
    return [float(x.split(":")[1].strip()) for x in vals]


loss_train, loss_val = [], []
iou_train, iou_val = [], []
with open(logfile, "r") as f:
    for line in f:
        if "train iou" not in line:
            continue
        vals = line.strip("\n").split(" | ")
        loss_train.append(vals[3])
        loss_val.append(vals[4])
        iou_train.append(vals[5])
        iou_val.append(vals[6])
if all([loss_train, loss_val, iou_train, iou_val]):
    loss_train = get_val(loss_train)
    loss_val = get_val(loss_val)
    iou_train = get_val(iou_train)
    iou_val = get_val(iou_val)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(loss_train, label="train")
    axs[0].plot(loss_val, label="val")
    # start, end = axs[0].get_ylim()
    # axs[0].set_yticks(arange(start, end + 0.1, 0.1))
    # axs[0].set_yticks(arange(0, 1.1, 0.1))
    axs[0].legend(loc="upper right")
    axs[0].title.set_text(
        f"train loss: {loss_train[-1]:.4f} | val loss: {loss_val[-1]:.4f}"
    )

    axs[1].plot(iou_train, label="train")
    axs[1].plot(iou_val, label="val")
    # start, end = axs[1].get_ylim()
    # axs[1].set_yticks(arange(start, end + 0.1, 0.1))
    axs[1].legend(loc="lower right")
    axs[1].title.set_text(
        f"train fwiou: {iou_train[-1]:.4f} | val fwiou: {iou_val[-1]:.4f}"
    )

    fig.suptitle(logfile.split("/")[-1], fontsize=16)
    plt.tight_layout()
    plt.savefig("logs/vis/%s.png" % logfilename)
else:
    print("no record found.")
