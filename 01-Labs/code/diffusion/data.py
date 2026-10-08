"""Load MNIST (28x28 handwritten digits) as tensors scaled to [-1, 1] and padded to 32x32."""
import io, os
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image

D = os.path.expanduser("~/.labs_data/mnist")

def load(split):
    cache = os.path.join(D, f"{split}.npz")
    if not os.path.exists(cache):
        t = pq.read_table(os.path.join(D, f"{split}.parquet")).to_pylist()
        x = np.stack([np.array(Image.open(io.BytesIO(r["image"]["bytes"]))) for r in t]).astype(np.uint8)
        y = np.array([r["label"] for r in t], dtype=np.int64)
        np.savez(cache, x=x, y=y)
    d = np.load(cache)
    x = torch.from_numpy(d["x"]).float().div(255).mul(2).sub(1)    # pixel values 0..255 -> -1..1
    x = F.pad(x, (2, 2, 2, 2), value=-1.0).unsqueeze(1)             # 28x28 -> 32x32, one channel
    return x, torch.from_numpy(d["y"])
