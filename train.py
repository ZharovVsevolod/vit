from vit import config
from vit.models import ViT_Lightning
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

def vit_train():
    dm = ...
    model = ViT_Lightning()

    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    wandb_log = WandbLogger(project="vit", name="vit_1", save_dir=config.BASE_OUTPUT)

    checkpoint = ...

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=wandb_log,
        callbacks=[checkpoint]
    )
    trainer.fit(model=model, datamodule=dm)

    wandb.finish()

if __name__ == "__main__":
    L.seed_everything(1702)
    vit_train()