import logging
import argparse
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.Neo_Model import Neo
from models.Neo_Model_valid import NeoValid
from models.Neo_Model_suffix_tree import NeoST
from models.Neo_Model_DP import NeoDP
from utils import MetricTracker

if __name__ == '__main__':
    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    # Init configs that are not given
    if 'seed' not in config:
        seed = 42
    if 'privacy_method' not in config:
        config.privacy_method = None
    if 'train_sets' not in config:
        config.train_sets = ""
    if 'valid_sets' not in config:
        config.valid_sets = []
    if 'valid_subset_path' not in config:
        config.valid_subset_path = None
    if 'valid_type_path' not in config:
        config.valid_type_path = None
    if 'learning_rate' not in config:
        config.learning_rate = 5e-5
    if 'negative_loss' not in config:
        config.negative_loss = True
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in config:
        config.num_train_epochs = 0
    if 'num_workers' not in config:
        config.num_workers = 0
    if 'wandb_log' not in config:
        config.wandb_log = False
    if 'strategy' not in config:
        config.strategy = None
    if 'fp16' not in config:
        config.fp16 = False
    if 'check_validation_only' not in config:
        config.check_validation_only = False
    if 'check_val_every_n_epoch' not in config:
        config.check_val_every_n_epoch = 1
    if 'tokenizer' not in config:
        config.tokenizer_name_or_path = config.model_name_or_path
    if 'target_length' not in config:
        config.target_length = None
    if 'el_n' not in config:
        config.el_n = [10]
    if 'el_threshold' not in config:
        config.el_threshold = 0
    if 'ma_threshold' not in config:
        config.ma_threshold = 0
    if 'min_train_epochs' not in config:
        config.min_train_epochs = 0
    if 'do_init_eval' not in config:
        config.do_init_eval = True if config.mode == 'unlearn' else False

    pl.seed_everything(seed, workers=True)

    # Set console logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s (%(filename)s:%(lineno)d) : %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set wandb logger
    if config.wandb_log:
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity='lklab_kaist')
    else:
        wandb_logger = None

    callbacks = [MetricTracker(config.wandb_run_name)]

    # Setting for pytorch lightning trainer
    train_params = dict(
        accumulate_grad_batches=config.gradient_accumulation_steps,
        accelerator='gpu',
        devices=config.ngpu,
        max_epochs=int(config.num_train_epochs),
        precision=16 if config.fp16 else 32,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_checkpointing=False,
        callbacks=callbacks,
        logger=wandb_logger,
        strategy=config.strategy,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        log_every_n_steps=1
    )

    if config.check_validation_only:
        trainer = pl.Trainer(**train_params)
        if config.privacy_method == 'dp':
            model = NeoDP(config)
        elif config.privacy_method == 'st':
            model = NeoST(config)
        else:
            model = NeoValid(config)
        model = NeoST(config)
        trainer.validate(model)
    else:
        trainer = pl.Trainer(**train_params)
        if config.do_init_eval:
            model = NeoValid(config)
            trainer.validate(model)
        model = Neo(config)
        trainer.fit(model)
