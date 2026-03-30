import argparse
import os
import sys

import torch
import wandb


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from DeepProtein.utils import generate_config


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {'true', 't', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', 'f', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def build_parser(description, default_target_encoding='CNN', default_epochs=100):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_encoding', type=str, default=default_target_encoding,
                        help='Encoding method for target proteins')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_proj', type=str, default='your_project_name', help='wandb project name')
    parser.add_argument('--lr', type=float, default=0.0001, help='0.0001/0.00001')
    parser.add_argument('--epochs', type=int, default=default_epochs, help='50/100')
    parser.add_argument('--compute_pos_enc', type=str2bool, default=False,
                        help='compute position encoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    return parser


def initialize_run(args, job_name):
    wandb.init(project=args.wandb_proj, name=job_name)
    wandb.config.update(vars(args))
    torch.manual_seed(args.seed)
    return os.getcwd()


def build_config(args, cls_hidden_dims, extra_config=None):
    config = generate_config(
        target_encoding=args.target_encoding,
        cls_hidden_dims=cls_hidden_dims,
        train_epoch=args.epochs,
        LR=args.lr,
        batch_size=args.batch_size,
    )
    if extra_config:
        config.update(extra_config)
    return config
