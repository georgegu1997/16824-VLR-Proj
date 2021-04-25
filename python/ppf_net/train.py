import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from ppf_net.models import getModel
from ppf_net.datasets import getDataloaders

@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    print("------------ CONFIG ------------")
    print(OmegaConf.to_yaml(config))

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    cwd = hydra.utils.get_original_cwd()

    # Get the model and dataloaders
    train_loader, valid_loader, test_loader = getDataloaders(config)
    model = getModel(config)

    trainer = pl.Trainer(
        default_root_dir=cwd,
        # gpus = config.train.gpus,
        # accelerator = 'ddp',
        # resume_from_checkpoint = config.resume_path,
        # logger=logger,
        # checkpoint_callback=checkpoint_callback,
        # max_epochs=config.model.max_epochs,
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
