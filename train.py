"""Train the small CNN on CIFAR-10 (CPU is enough: a few minutes per epoch at most)."""
import argparse
import time

import torch
import torch.nn as nn

from common import CKPT, MODEL_DIR, SmallResNet, load_cifar10


def augment(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random crop (4px reflect padding) and horizontal flip."""
    b = x.shape[0]
    xp = nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
    ar = torch.arange(32)
    rows = torch.randint(0, 9, (b, 1)) + ar
    cols = torch.randint(0, 9, (b, 1)) + ar
    flip = torch.rand(b, 1) < 0.5
    cols = torch.where(flip, cols.flip(1), cols)
    return xp[
        torch.arange(b)[:, None, None, None],
        torch.arange(3)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i : i + bs]).argmax(1) == y[i : i + bs]).sum().item()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x_tr, y_tr = (torch.from_numpy(a) for a in load_cifar10(train=True))
    x_te, y_te = (torch.from_numpy(a) for a in load_cifar10(train=False))

    model = SmallResNet()
    steps = args.epochs * (len(x_tr) // args.batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(x_tr))
        running = 0.0
        n_batches = len(x_tr) // args.batch_size
        for b in range(n_batches):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            loss = loss_fn(model(augment(x_tr[idx])), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        acc = evaluate(model, x_te, y_te)
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {running / n_batches:.3f}  "
              f"test acc {acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")


if __name__ == "__main__":
    main()
