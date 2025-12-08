import os
#!/usr/bin/env python3
"""
Minimal feature extractor using Activeloop Deeplake.
"""
import argparse
import io
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# deeplake loader
def load_deeplake_dataset(hub_uri):
    """
    Minimal loader that uses deeplake.open (v4) or falls back to deeplake.load (v3).
    Assumes deeplake is installed via requirements.txt. Raises on failure.
    Returns list of (PIL.Image, label, pseudo-path).
    """
    import deeplake, io, numpy as _np

    # prefer modern API, fall back to v3
    if hasattr(deeplake, "open"):
        ds = deeplake.open(hub_uri)
    else:
        ds = deeplake.load(hub_uri)

    img_field = next((n for n in ("images", "image", "img") if n in ds), None)
    label_field = next((n for n in ("labels", "label", "y") if n in ds), None)
    if img_field is None:
        raise RuntimeError(f"No image field found in deeplake dataset {hub_uri}")

    items = []
    for idx, sample in enumerate(ds):
        entry = sample[img_field]
        # try numpy-like array, then bytes, then generic array
        pil = None
        try:
            arr = entry.numpy()
            pil = Image.fromarray(arr.astype("uint8"))
        except Exception:
            pass

        if pil is None:
            try:
                pil = Image.open(io.BytesIO(bytes(entry))).convert("RGB")
            except Exception:
                arr = _np.array(entry)
                pil = Image.fromarray(arr.astype("uint8"))

        lbl = "unknown"
        if label_field is not None:
            try:
                raw = sample[label_field]
                try:
                    raw = raw.numpy()
                except Exception:
                    pass
                if isinstance(raw, (bytes, bytearray)):
                    lbl = raw.decode("utf-8")
                else:
                    try:
                        lbl = str(raw.item())
                    except Exception:
                        lbl = str(raw)
            except Exception:
                lbl = "unknown"

        items.append((pil, lbl, f"deeplake://{hub_uri}/{idx}"))

    return items

class DeeplakeFaceDataset(Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pil, label, path = self.items[i]
        img = pil
        if self.tfm:
            img = self.tfm(img)
        return img, label, path

# model / utils
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def build_model(device):
    print("[INFO] loading pretrained resnet50")
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract(model, dl, device):
    embs = []
    labels = []
    paths = []
    for imgs, lbs, ps in tqdm(dl, desc="extract"):
        imgs = imgs.to(device)
        feats = model(imgs).cpu().numpy()
        embs.append(feats)
        labels.extend(lbs)
        paths.extend(ps)
    if embs:
        embs = np.vstack(embs).astype("float32")
    else:
        embs = np.zeros((0,2048), dtype="float32")
    return embs, labels, paths

def save(out, embs, labels, paths):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=embs, labels=np.array(labels), paths=np.array(paths))
    print(f"[INFO] saved {embs.shape[0]} embeddings to {out}")

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deeplake-uri", default="hub://activeloop/lfw",
                   help="Deeplake hub uri (e.g. hub://activeloop/lfw)")
    p.add_argument("--output", default="data/embeddings/lfw_deeplake_embeddings.npz")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    tfm = get_transform()

    items = load_deeplake_dataset(args.deeplake_uri)
    ds = DeeplakeFaceDataset(items, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    model = build_model(device)
    embs, labels, paths = extract(model, dl, device)
    save(args.output, embs, labels, paths)

if __name__ == "__main__":
    main()