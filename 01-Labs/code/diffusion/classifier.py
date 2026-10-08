"""A small CNN digit classifier. It is only used as a judge: does a generated image look like the digit we asked for?"""
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load
from common import dev, RUN

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))

def load_classifier():
    c = Classifier().to(dev)
    c.load_state_dict(torch.load(os.path.join(RUN, "classifier.pt"), map_location=dev))
    return c.eval()

if __name__ == "__main__":
    torch.manual_seed(0)
    x, y = load("train"); xt, yt = load("test")
    c = Classifier().to(dev); opt = torch.optim.Adam(c.parameters(), 1e-3); t0 = time.time()
    for epoch in range(2):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(c(x[b].to(dev)), y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    c.eval()
    with torch.no_grad():
        acc = (c(xt.to(dev)).argmax(1).cpu() == yt).float().mean().item()
    torch.save(c.state_dict(), os.path.join(RUN, "classifier.pt"))
    print(f"classifier trained for 2 epochs in {time.time()-t0:.0f}s, accuracy on 10,000 real test digits: {acc:.1%}")
