import argparse
import logging
import time
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def _save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


# def _save_checkpoint(model, optimizer, epoch, loss, args):
#     print("epoch: {} - loss: {}".format(epoch+1, loss))
#     checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
#     print("Saving the Checkpoint: {}".format(checkpointing_path))
#     torch.save({
#         'epoch': epoch+1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         }, checkpointing_path)

    
# def _load_checkpoint(model, optimizer, args):
#     print("--------------------------------------------")
#     print("Checkpoint file found!")
#     print("Loading Checkpoint From: {}".format(args.checkpoint_path + '/checkpoint.ckpt'))
#     checkpoint = torch.load(args.checkpoint_path + '/checkpoint.ckpt')
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_states'])
#     epoch_number = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
#     print('Resuming training from epoch: {}'.format(epoch_number+1))
#     print("--------------------------------------------")
#     return model, optimizer, epoch_number

    
def model_fn(model_dir):
    print('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

class create_resnet_model(nn.Module):
    def __init__(self):
        super(create_resnet_model, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 10)


    def forward(self, x):
        out = self.model(x)
        return out
    
class model_related_functions(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_resnet_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // args.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
def main_train_loop(args):
    
    print("Loading Cifar10 dataset.....", end='')
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    
    print('data loading done')

    print("Loading model....." , end='')
    model = model_related_functions()
    model.datamodule = cifar10_dm
    print('model loading done')


    # training the model
    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=args.epochs,
        gpus=args.num_gpus,
        logger=TensorBoardLogger("lightning_logs/", name="resnet"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        strategy="dp",

    )
    trainer.fit(model, cifar10_dm)
    _save_model(model,args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=4, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()
    
    print('checking arguments ', args)
    
    main_train_loop(args)