import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
from pathlib import Path
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from numpy import cos, sin

DEVICE = 0
OUTDIR = "."
BATCH_SIZE = 32
facet = os.path.basename(os.path.abspath(os.path.join(os.getcwd(), "..")))

train = torch.load(f"../../../dataset/{facet}/train_dataset.pth.tar")
val = torch.load(f"../../../dataset/{facet}/val_dataset.pth.tar")
test = torch.load(f"../../../dataset/{facet}/test_dataset.pth.tar")

modelparams = {
    "feat_dim": 128,
    "activation": "swish",
    "n_rbf": 20,
    "cutoff": 5.0,
    "num_conv": 3,
    "output_keys": ["energy"],
    "grad_keys": ["energy_grad"],
    "skip_connection": {"energy": False},
    "learnable_k": False,
    "conv_dropout": 0.0,
    "readout_dropout": 0.0,
    "means": {"energy": train.props["energy"].mean().item()},
    "stddevs": {"energy": train.props["energy"].std().item()},
}

if os.path.isfile("best_model"):
    model = torch.load("best_model")
else:
    model = get_model(modelparams, model_type="Painn")

original_model = copy.deepcopy(model)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, collate_fn=collate_dicts, sampler=RandomSampler(train))
val_loader = DataLoader(val, batch_size=BATCH_SIZE, collate_fn=collate_dicts)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=collate_dicts)

loss_fn = loss.build_mse_loss(loss_coef={"energy_grad": 0.95, "energy": 0.05})

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = Adam(trainable_params, lr=3e-4)

train_metrics = [metrics.MeanAbsoluteError("energy"), metrics.MeanAbsoluteError("energy_grad")]

train_hooks = [
    hooks.MaxEpochHook(5000),
    hooks.CSVHook(
        OUTDIR,
        metrics=train_metrics,
    ),
    hooks.PrintingHook(OUTDIR, metrics=train_metrics, separator=" | ", time_strf="%M:%S"),
    hooks.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=50,
        factor=0.5,
        min_lr=1e-6,
        window_length=1,
        stop_after_min=True,
    ),
]

T = Trainer(
    model_path=OUTDIR,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    checkpoint_interval=1,
    hooks=train_hooks,
    mini_batches=1,
)


T.train(device=DEVICE, n_epochs=5000)

def make_rot(alpha, beta, gamma):
    r = torch.Tensor(
        [
            [
                cos(alpha) * cos(beta),
                cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
                cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
            ],
            [
                sin(alpha) * cos(beta),
                sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
                sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
            ],
            [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)],
        ]
    )

    return r


r = make_rot(0.2, 0.1, 0.4)
print(torch.matmul(r, r.transpose(0, 1)))

nxyz = train.props["nxyz"][0]
rots = [torch.diag(torch.ones(3)), make_rot(1.4, -0.5, 1.3)]
original_model.to(DEVICE)

for rot in rots:
    xyz = torch.stack([torch.matmul(rot, i[1:]) for i in nxyz])
    z = nxyz[:, 0].reshape(-1, 1)
    this_nxyz = torch.cat([z, xyz], dim=-1).to(DEVICE)
    batch = {
        "nxyz": this_nxyz,
        "num_atoms": torch.LongTensor([len(nxyz)]),
        "nbr_list": train.props["nbr_list"][0].to(DEVICE),
    }
    results = original_model(batch)
    energy = results["energy"].cpu()

    print("%.8f " % energy.item())

    energy_grad = results["energy_grad"].cpu()

    rot_grad = torch.stack([torch.matmul(rot.transpose(0, 1), i) for i in energy_grad])
    print(rot_grad)

results, targets, val_loss = evaluate(T.get_best_model(), test_loader, loss_fn, device=DEVICE)

units = {"energy_grad": r"kcal/mol/$\AA$", "energy": "kcal/mol"}

fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6))

for ax, key in zip(ax_fig, units.keys()):
    pred_fn = torch.cat
    targ_fn = torch.cat
    if all([len(i.shape) == 0 for i in results[key]]):
        pred_fn = torch.stack
    if all([len(i.shape) == 0 for i in targets[key]]):
        targ_fn = torch.stack

    pred = pred_fn(results[key], dim=0).view(-1).detach().cpu().numpy()
    targ = targ_fn(targets[key], dim=0).view(-1).detach().cpu().numpy()

    mae = abs(pred - targ).mean()

    ax.hexbin(pred, targ, mincnt=1)

    lim_min = min(np.min(pred), np.min(targ)) * 1.1
    lim_max = max(np.max(pred), np.max(targ)) * 1.1

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")

    ax.plot((lim_min, lim_max), (lim_min, lim_max), color="#000000", zorder=-1, linewidth=0.5)

    ax.set_title(key.upper(), fontsize=14)
    ax.set_xlabel("predicted %s (%s)" % (key, units[key]), fontsize=12)
    ax.set_ylabel("target %s (%s)" % (key, units[key]), fontsize=12)
    ax.text(0.1, 0.9, "MAE: %.2f %s" % (mae, units[key]), transform=ax.transAxes, fontsize=14)

    print(f'{facet} {key} MAE: {mae}')
plt.savefig('./test_dataset.png')
