"""SimCLR training on CIFAR10."""

import lightning as pl
import torch
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
import sys
from pathlib import Path
import os 
sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir


def worker_init_fn(worker_id):
    # Set CPU affinity when available (Linux); noop on macOS/others
    if hasattr(os, "sched_setaffinity"):
        try:
            n_cpus = os.cpu_count() or 1
            os.sched_setaffinity(0, range(n_cpus)) # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            pass

class ProtoCLRModule(nn.Module):
    def __init__(self, num_prototypes: int, embedding_dim: int, proto_processor: torch.nn.Module = None):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        self.proto_processor = proto_processor or nn.Identity()
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.proto_processor(self.prototypes[indices])


def protoclr_forward(self, batch, stage: str):
    """
    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head mapping features to latent space
            - simclr_loss: NT-Xent contrastive loss function
            - protoclr: ProtoCLRModule instance for prototype handling
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': NT-Xent contrastive loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the SimCLR paper :cite:`chen2020simple`.
    """
    out = {}

    views = forward._get_views_list(batch)
    if views is not None:
        if len(views) != 1:
            raise ValueError(
                f"ProtoCLR requires exactly 1 view, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )
        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)


        # Concatenate labels for callbacks (probes need this)
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            proto_embeddings = [self.protoclr(view["sample_idx"]) for view in views]

            out["loss"] = self.simclr_loss(projections[0], proto_embeddings[0])
            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


suffix = ""

protoclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)
suffix = f"{suffix}"


# protoclr_transform = transforms.MultiViewTransform(
#     [
#         transforms.Compose(
#             transforms.RGB(),
#             transforms.ToImage(**spt.data.static.CIFAR10),
#         ),
#     ]
# )
# suffix = f"{suffix}-noaug"

# protoclr_transform = transforms.MultiViewTransform(
#     [
#         transforms.Compose(
#             transforms.RGB(),
#             # transforms.RandomResizedCrop((32, 32)),
#             # transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(
#                 brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ToImage(**spt.data.static.CIFAR10),
#         ),
#     ]
# )
# suffix = f"{suffix}-grayjitter"



val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)


ds = "cifar10"
ds_cls = torchvision.datasets.CIFAR10
ds_num_cls = 10

# ds = "cifar100"
# ds_cls = torchvision.datasets.CIFAR100
# ds_num_cls = 100


data_dir = get_data_dir(ds)
cifar_train = ds_cls(
    root=str(data_dir), train=True, download=True
)
cifar_val = ds_cls(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],
    transform=protoclr_transform,
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=val_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=256,
    num_workers=8,
    drop_last=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
    worker_init_fn=worker_init_fn,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

backbone = spt.backbone.from_torchvision(
    "resnet18",
    low_resolution=True,
)
backbone.fc = torch.nn.Identity()


projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

proto_emb_dim = 32
hidden = 512
proto_processor = nn.Sequential(
    nn.Linear(proto_emb_dim, hidden),
    nn.BatchNorm1d(hidden),
    nn.ReLU(inplace=True),
    # nn.Linear(2048, 2048),
    # nn.BatchNorm1d(2048),
    # nn.ReLU(inplace=True),
    nn.Linear(hidden, 256),
)
suffix = f"-profactv2-{proto_emb_dim}-hid{hidden}"

protoclr = ProtoCLRModule(num_prototypes=len(train_dataset), embedding_dim=proto_emb_dim, proto_processor=proto_processor)

# projector = nn.Identity()
# protoclr = ProtoCLRModule(num_prototypes=len(train_dataset), embedding_dim=512)
# suffix = f"{suffix}-no-proj"


loss_name = "infonce"
# loss_name = "ntxent"
# loss_name= "clip"

if loss_name == "infonce":
    loss_cls = spt.losses.InfoNCELoss
elif loss_name == "ntxent":
    loss_cls = spt.losses.NTXEntLoss
elif loss_name == "clip":
    loss_cls = spt.losses.CLIPLoss
else:
    assert False, loss_name



module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=protoclr_forward,
    protoclr=protoclr,
    simclr_loss=loss_cls(temperature=0.5),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    module,
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, ds_num_cls),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(ds_num_cls),
        "top5": torchmetrics.classification.MulticlassAccuracy(ds_num_cls, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(ds_num_cls)},
    input_dim=512,
    k=10,
)

exp_name = f"protoclr-resnet18-{ds}-{loss_name}{suffix}"

wandb_logger = WandbLogger(
    group=exp_name,
    entity="mprzewie",
    project="spt-prototypes",
    log_model=False,
)

# Create learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    max_epochs=1000,
    num_sanity_val_steps=0,
    callbacks=[knn_probe, linear_probe, lr_monitor],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    default_root_dir=f"outputs/{exp_name}",
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
