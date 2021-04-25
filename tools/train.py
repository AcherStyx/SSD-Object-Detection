import os
import json

import tensorflow as tf
from models import SSDObjectDetectionModel
from data_loaders import SSDDataLoader

import yaml
import argparse


def load_config(yaml_file: str):
    with open(yaml_file, "r") as f:
        config = yaml.load(f)
    return config


def train(config):
    data = SSDDataLoader(dataset_root=config["data"]["dataset_root"],
                         shuffle_buffer=config["data"]["shuffle_buffer"],
                         dataset=config["data"]["dataset"])
    model = SSDObjectDetectionModel(classes=config["data"]["num_classes"],
                                    log_dir=config["model"]["log_dir"])

    # scheduler->optimizer
    if config["model"]["train"]["optimizer"].lower() == "adam":
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["model"]["train"]["lr"]["initial"],
            decay_steps=config["model"]["train"]["lr"]["decay_step"],
            decay_rate=config["model"]["train"]["lr"]["decay_rate"]
        )
        warmup_lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config["model"]["warmup"]["lr"]["lr_start"],
            decay_steps=config["model"]["warmup"]["lr"]["step"],
            end_learning_rate=config["model"]["warmup"]["lr"]["lr_end"]
        )
        optimizer = tf.keras.optimizers.Adam(lr_scheduler)
        warmup_optimizer = tf.keras.optimizers.Adam(warmup_lr_scheduler)
    else:
        raise ValueError

    with open(os.path.join(model.get_log_dir(), "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4, separators=(',', ':'))
    with model.get_log_writer().as_default(step=0):
        tf.summary.text("config", str(config))

    model.train(data_loader=data,
                epoch=config["model"]["train"]["epoch"],
                batch_size=config["model"]["train"]["batch_size"],
                optimizer=optimizer,
                warmup=config["model"]["warmup"],
                warmup_optimizer=warmup_optimizer,
                warmup_step=config["model"]["warmup"]["lr"]["step"],
                visualization_log_interval=config["model"]["log_interval"])
    model.save(os.path.join(model.get_log_dir(), config["model"]["save"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train ssd model")
    parser.add_argument("config", type=str, help="yaml config file")

    args = parser.parse_args()
    yaml_config = load_config(args.config)

    train(yaml_config)
