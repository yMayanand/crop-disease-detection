from imports import *
from data import *

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(num_classes=39)
        self.accuracy = tm.Accuracy()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        return out
        
    def training_step(self, batch, batch_idx):
        data, label = batch
        out = self(data)
        acc = self.accuracy(out, label)
        loss = self.loss_func(out, label)
        self.log('Loss/train', loss)
        self.log('Accuracy/train', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        out = self(data)
        acc = self.accuracy(out, label)
        loss = self.loss_func(out, label)
        self.log('Loss/val', loss)
        self.log('Accuracy/val', acc)
        return loss

    def predict_step(self, batch, batch_idx):
        data, label = batch
        out = self(data)
        _, idx = torch.max(out, dim=1)
        return idx

    def test_step(self, batch, batch_idx):
        data, label = batch
        out = self(data)
        acc = self.accuracy(out, label)
        loss = self.loss_func(out, label)
        self.log('Loss/test', loss)
        self.log('Accuracy/test', acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=3e-3, weight_decay=0) # wd = 3e-3
        steps_per_epoch = math.ceil(len(train_ds)/64)
        scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(optimizer, 
                                                       max_lr=1e-2, epochs=10, 
                                                       steps_per_epoch=steps_per_epoch),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=64, num_workers=2)
        return train_dl

    def val_dataloader(self):
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=2)
        return val_dl

    def test_dataloader(self):
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, num_workers=2)
        return test_dl

model = Model()

logger = pl.loggers.TensorBoardLogger('lightning_logs', name='baseline-resnet18')
trainer = pl.Trainer(gpus=1, logger=logger, 
                     max_epochs=10, profiler='simple',
                     callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="Loss/val"),
                                LearningRateMonitor("step")])

trainer.fit(model)