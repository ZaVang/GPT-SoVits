import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Example with modularized command line arguments.')

    # General arguments
    general = parser.add_argument_group('General')
    general.add_argument('-c', '--config', type=str, default='src/configs/sovits.json', help='JSON file for configuration')
    general.add_argument('-n', '--name', type=str, required=True, help='speaker name')
    general.add_argument('-t', '--task_type', type=str, default="sovits", choices=["sovits", "gpt"], help='task type')

    # Training arguments
    training_args = parser.add_argument_group('Training')
    training_args.add_argument('-e', '--epochs', type=int, default=20, help='finetune epochs')
    training_args.add_argument('-lr', '--learning_rate', type=float, default=0.4, help='text low learning rate')
    training_args.add_argument('-bs', '--batch_size', type=int, default=16, help='batch size')
    training_args.add_argument('-pd', '--pretrained_D', type=str, default='pretrained_models/sovits_weights/pretrained/s2D488k.pth', help='pretrained D path')
    training_args.add_argument('-pg', '--pretrained_G', type=str, default='pretrained_models/sovits_weights/pretrained/s2G488k.pth', help='pretrained G path')
    training_args.add_argument('-nw', '--num_workers', type=int, default=0, help='num workers for dataloader')
    
    # Saving/Logging arguments
    save_log_args = parser.add_argument_group('Saving/Logging')
    save_log_args.add_argument('-d', '--model_dir', type=str, default="pretrained_models", help='Model directory')
    save_log_args.add_argument('-l', '--log_dir', type=str, default="logs/", help='Log directory')
    save_log_args.add_argument('--if_save_latest', action='store_false', help='Save the latest weights')
    save_log_args.add_argument('--if_save_every_weights', action='store_false', help='Save every weights')
    save_log_args.add_argument('--save_every_epoch', type=int, default=5, help='Save every n epoch')
    save_log_args.add_argument('--keep_ckpts', type=int, default=5, help='Keep the last n checkpoints')
    
    args = parser.parse_args()
    return args


def modify_config(config, args):
    config["speaker_name"] = args.name
    config["train"]["epochs"] = args.epochs
    config["train"]["text_low_lr_rate"] = args.learning_rate
    config["train"]["batch_size"] = args.batch_size
    config["train"]["num_workers"] = args.num_workers
    

def get_hparams(init=True):
    args = get_args()
    model_dir = os.path.join(args.model_dir, f"{args.task_type}_weights", args.name)
    save_dir = os.path.join(args.log_dir, args.name, args.task_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_path = args.config
    config_save_path = os.path.join(save_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)
    
    modify_config(config, args)
    
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
    
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.save_dir = save_dir
    hparams.train.pretrained_D = args.pretrained_D
    hparams.train.pretrained_G = args.pretrained_G
    hparams.train.if_save_latest = args.if_save_latest
    hparams.train.if_save_every_weights = args.if_save_every_weights
    hparams.train.save_every_epoch = args.save_every_epoch
    hparams.train.keep_ckpts = args.keep_ckpts
    hparams.log_root_dir = args.log_dir
    hparams.data.exp_dir = os.path.join(args.log_dir, args.name)
    # hparams.resume_from_ckptD = args.resume_from_ckptD
    # hparams.resume_from_ckptG = args.resume_from_ckptG
    return hparams


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
    

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")