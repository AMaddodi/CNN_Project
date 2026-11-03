import argparse
import torch
from PIL import Image

from src.data import get_transforms
from src.model import CLASSES, CNN

def load_model(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location=device)
    model = CNN(**ck["config"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()

@torch.no_grad()
def predict(model, image: Image.Image, device, top_k: int =3):
    """ Rescale the image as per transform function """
    transform = get_transforms(False)(image.convert("RGB").resize((32,32))).unsqueeze(0).to(device)
    probs = torch.softmax(model(transform), dim=1)[0]
    vals, idx =probs.topk(top_k)
    return [(CLASSES[i], val.item()) for val, i in zip(vals, idx)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default="models/best.pt")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, p in predict(load_model(args.checkpoint, dev), Image.open(args.image), dev):
        print(f"{name:6s}  {p:.1%}")