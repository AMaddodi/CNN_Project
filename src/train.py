import json
import random
import configparser
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.data import get_loader
from src.model import CNN, count_parameters

def set_seed(seed: int):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.seed()
    torch.cuda.manual_seed_all(seed)
    

@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0,0,0
    for image, lable in loader:
        image, lable = image.to(device), lable.to(device)
        out = model(image)
        loss_sum = criterion(out, lable).item() * lable.size(0)
        correct += (out.argmax(1)==lable).sum().item()
        total += lable.size(0)
    return loss_sum / total, 100.0 * correct / total


def build_optimiser(cfg, model):
    if cfg["optimizer"]  == 'sgd':
        return torch.optim.SGD(model.parameters(), lr = cfg['lr'], momentum=0.9,
                               weight_decay=cfg['weight_decay'], nesterov=True)
    return torch.optim.AdamW(model.parameters(), lr = cfg['lr'],
                             weight_decay=cfg['weight_decay'])


def read_config():
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    cfg = {'seed': config.getint('general', 'seed'), 'data_root': config.get('general', 'data_root'),
           'batch_size': config.getint('general', 'batch_size'), 'val_size': config.getint('general', 'val_size'),
           'augment': config.getboolean('general', 'augment'), 'num_workers': config.getint('general', 'num_workers'),
           'epochs': config.getint('training', 'epochs'), 'optimizer': config.get('training', 'optimizer'),
           'lr': config.getfloat('training', 'lr'), 'weight_decay': config.getfloat('training', 'weight_decay'),
           'scheduler': config.get('training', 'scheduler'), 'output_dir': config.get('training', 'output_dir'),
           'checkpoint': config.get('training', 'checkpoint'), 'model': {
            'p_drop': config.getfloat('model', 'p_drop'),
            'hidden': config.getint('model', 'hidden')
        }}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="run")
    args = ap.parse_args()

    cfg = read_config()
    print(cfg)
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    train_loader, val_loader, _ = get_loader(cfg['data_root'],cfg['batch_size'],cfg['val_size'],
                                             cfg['augment'],cfg['num_workers'],cfg['seed'])

    model = CNN(**cfg['model']).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimiser(cfg, model)
    scheduler = None
    if cfg['scheduler'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg['lr'], epochs=cfg['epochs'],
            steps_per_epoch=len(train_loader)
        )
    history = {'train_loss': [], 'val_loss': [], 'val_acc':[]}
    best_acc = 0.0
    ckpt = Path(cfg['checkpoint'])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg['epochs']+1):
        model.train()
        running, n = 0.0, 0
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(model(image),label)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running += loss.item() * label.size(0)
            n += label.size(0)

        val_loss , val_acc = evaluate_loader(model, val_loader, criterion, device)
        history['train_loss'].append(running/n)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"epoch {epoch:02d}/{cfg['epochs']}  train_loss={running/n:.4f} "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "config": cfg,
                        "val_acc": val_acc, "epoch": epoch}, ckpt)

    with open(out_dir / f"{args.run_name}_history.json", "w") as f:
        json.dump({"best_val_acc": best_acc, "history": history, "config": cfg}, f, indent=2)
    print(f"Best val accuracy: {best_acc:.2f}%  (checkpoint: {ckpt})")


if __name__ == "__main__":
    main()

