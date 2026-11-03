import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import denormalize, get_loader
from src.model import CNN, CLASSES


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    imgs, ys, preds = [], [], []
    for x, y in loader:
        out = model(x.to(device)).cpu()
        imgs.append(x)
        ys.append(y)
        preds.append(out.argmax(1))
    return torch.cat(imgs), torch.cat(ys).numpy(), torch.cat(preds).numpy()


def plot_curves(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="Epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(history["val_acc"], color="tab:green")
    ax[1].set(title="Validation accuracy (%)", xlabel="Epoch")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion(y, p, n=10):
    cm = np.zeros((n, n), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    return cm


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix (test set)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(acc, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(acc)
    ax.barh([CLASSES[i] for i in order], acc[order], color="tab:blue")
    ax.set(xlabel="Accuracy (%)", title="Per-class test accuracy")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(imgs, y, p, path, n=15):
    wrong = np.where(y != p)[0][:n]
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.5))
    for ax, idx in zip(axes.flat, wrong):
        ax.imshow(denormalize(imgs[idx]))
        ax.set_title(f"pred: {CLASSES[p[idx]]}\ntrue: {CLASSES[y[idx]]}",
                     fontsize=9, color="red")
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best.pt")
    ap.add_argument("--history", default="results/run_history.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.checkpoint, map_location=device)
    cfg = ck["config"]
    model = CNN(**cfg["model"]).to(device)
    model.load_state_dict(ck["model"])

    _, _, test_loader = get_loader(cfg["data_root"], cfg["batch_size"], cfg["val_size"],
                                    False, cfg["num_workers"], cfg["seed"])
    imgs, y, p = collect_predictions(model, test_loader, device)

    cm = confusion(y, p)
    per_class = 100 * cm.diagonal() / cm.sum(1)
    overall = 100 * (y == p).mean()
    print(f"Test accuracy: {overall:.2f}%")
    for c, a in zip(CLASSES, per_class):
        print(f"  {c:6s} {a:.1f}%")

    # Most-confused pairs (off-diagonal)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], CLASSES[i], CLASSES[j])
                    for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:5]

    with open(out / "metrics.json", "w") as f:
        json.dump({"test_accuracy": float(overall),
                   "per_class_accuracy": {c: float(a) for c, a in zip(CLASSES, per_class)},
                   "top_confusions": [{"count": int(c), "true": t, "predicted": q}
                                      for c, t, q in pairs]}, f, indent=2)

    if Path(args.history).exists():
        plot_curves(json.load(open(args.history))["history"], out / "training_curves.png")
    plot_confusion(cm, out / "confusion_matrix.png")
    plot_per_class(per_class, out / "per_class_accuracy.png")
    plot_misclassified(imgs, y, p, out / "misclassified.png")
    print(f"Saved metrics and plots to {out}/")


if __name__ == "__main__":
    main()
