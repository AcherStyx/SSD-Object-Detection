import os
import json
import logging
import yaml
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from models import SSDObjectDetectionModel
from data_loaders import SSDDataLoader

logger = logging.getLogger(__name__)


def load_config(yaml_file: str):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    data = SSDDataLoader(
        dataset_root=config["data"]["dataset_root"],
        shuffle=config["data"]["shuffle"],
        dataset=config["data"]["dataset"],
        mini_batch=config["data"]["mini_batch"]["num_data"] if config["data"]["mini_batch"]["enable"] else 0)
    model = SSDObjectDetectionModel(classes=config["data"]["num_classes"],
                                    log_dir=config["model"]["log_dir"])
    # scheduler
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config["model"]["train"]["lr"]["initial"],
        decay_steps=config["model"]["train"]["lr"]["decay_step"],
        decay_rate=config["model"]["train"]["lr"]["decay_rate"]
    )
    warmup_lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config["model"]["warmup"]["lr"]["start"],
        decay_steps=config["model"]["warmup"]["step"],
        end_learning_rate=config["model"]["warmup"]["lr"]["end"]
    )
    # optimizer
    if config["model"]["train"]["optimizer"]["name"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(lr_scheduler, **config["model"]["train"]["optimizer"])
    elif config["model"]["train"]["optimizer"]["name"].lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr_scheduler, **config["model"]["train"]["optimizer"])
    else:
        raise ValueError
    if config["model"]["warmup"]["optimizer"]["name"].lower() == "adam":
        warmup_optimizer = tf.keras.optimizers.Adam(warmup_lr_scheduler, **config["model"]["warmup"]["optimizer"])
    elif config["model"]["warmup"]["optimizer"]["name"].lower() == "sgd":
        warmup_optimizer = tf.keras.optimizers.SGD(warmup_lr_scheduler, **config["model"]["warmup"]["optimizer"])
    else:
        raise ValueError

    with open(os.path.join(model.get_log_dir(), "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4, separators=(',', ':'))
    with model.get_log_writer().as_default(step=0):
        tf.summary.text("config", str(config))

    model.train(data_loader=data,
                cfg=SSDObjectDetectionModel.TrainConfig(epoch=config["model"]["train"]["epoch"],
                                                        batch_size=config["model"]["train"]["batch_size"],
                                                        optimizer=optimizer,
                                                        warmup=config["model"]["warmup"]["enable"],
                                                        warmup_optimizer=warmup_optimizer,
                                                        warmup_step=config["model"]["warmup"]["step"],
                                                        visualization_log_interval=config["model"]["log_interval"],
                                                        split_batch=config["model"]["split_train"]["enable"],
                                                        split_batch_size=config["model"]["split_train"]["batch_size"]))
    model.save(os.path.join(model.get_log_dir(), config["model"]["save"]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="train ssd model")
    parser.add_argument("config", type=str, help="yaml config file")

    args = parser.parse_args()
    yaml_config = load_config(args.config)

    train(yaml_config)
