import argparse

import numpy as np

from config import cfg
from engine import Trainer
from model import Model
from tfflat.utils import mem_info


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=str, dest='gpu_ids')
  parser.add_argument('--weights', type=str, dest='weights', default='none')
  args = parser.parse_args()

  if not args.gpu_ids:
    args.gpu_ids = str(np.argmin(mem_info()))
  return args


args = parse_args()
cfg.set_args(args.gpu_ids)
trainer = Trainer(Model(), cfg)
trainer.train(args.weights)
