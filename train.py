import os
import argparse
import datetime
from omegaconf import OmegaConf
from utils import instantiate_from_config
import pytorch_lightning as L
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader


class ConfigSaveCallback(Callback):
    def __init__(self, configs: dict):
        self.configs = configs

    def on_fit_start(self, trainer, pl_module):
        logger = trainer.logger
        save_dir = os.path.join(logger.log_dir, logger.name, logger.version, "configs")
        output_file = "config.yaml"
        # OmegaConf.save(
        #     config=OmegaConf.create(self.configs), f=os.path.join(save_dir, output_file)
        # )


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/policy.yaml")
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.config:
        name = opt.config.split("/")[-1].split(".")[0]
    else:
        raise ValueError("config must be specified")

    configs = [OmegaConf.load(opt.config)]
    cli = OmegaConf.from_dotlist(unknown)
    configs = OmegaConf.merge(*configs, cli)

    model_configs = configs.get("model", OmegaConf.create())
    trainer_configs = configs.get("trainer", OmegaConf.create())
    dataset_configs = trainer_configs.get("dataset", OmegaConf.create())

    logger_configs = trainer_configs.get("logger", OmegaConf.create())
    logger_configs["params"]["name"] = name
    logger_configs["params"]["version"] = now
    ckptdir = os.path.join(
        logger_configs["params"]["save_dir"],
        logger_configs["params"]["name"],
        logger_configs["params"]["version"],
        "checkpoints",
    )
    default_callback_config = {
        "checkpoint_callback_configs": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "checkpoints-{epoch:03}",
                "verbose": True,
                "monitor": trainer_configs["monitor"],
                "save_top_k": 3,
                "save_weights_only": True,
            },
        },
        "save_configs_callback_configs": {
            "target": "train.ConfigSaveCallback",
            "params": {"configs": configs},
        },
    }

    # trainer config
    trainer_kwargs = {
        "logger": instantiate_from_config(logger_configs),
        "callbacks": [
            instantiate_from_config(default_callback_config[k])
            for k in default_callback_config.keys()
        ],
    }
    trainer_configs["accelerator"] = "gpu"
    trainer_configs["strategy"] = "ddp"

    if not "gpus" in trainer_configs:
        del trainer_configs["gpus"]
        del trainer_configs["strategy"]
        print("using cpu")
    else:
        gpuinfo = trainer_configs["gpus"]
        print(f"using gpu: {gpuinfo}")
    trainer_opt = argparse.Namespace(**trainer_configs)
    dataset = instantiate_from_config(dataset_configs)
    dataloader = DataLoader(
        dataset, **(trainer_configs.get("dataloader", OmegaConf.create()))
    )
    # initialize model
    model = instantiate_from_config(model_configs)
    # initialize dataset

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.fit(model, dataloader)
