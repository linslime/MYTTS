import os
import sys
import argparse
import logging
import json
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def load_wav_to_torch(full_path):
	sampling_rate, data = read(full_path)
	return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
	with open(filename, encoding='utf-8') as f:
		filepaths_and_text = [line.strip().split(split) for line in f]
	return filepaths_and_text


def get_hparams(init=True):
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
	                    help='JSON file for configuration')
	parser.add_argument('-m', '--model', type=str, required=True,
	                    help='Model name')
	
	args = parser.parse_args()
	model_dir = os.path.join("./logs", args.model)
	
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	
	config_path = args.config
	config_save_path = os.path.join(model_dir, "config.json")
	if init:
		with open(config_path, "r") as f:
			data = f.read()
		with open(config_save_path, "w") as f:
			f.write(data)
	else:
		with open(config_save_path, "r") as f:
			data = f.read()
	config = json.loads(data)
	
	hparams = HParams(**config)
	hparams.model_dir = model_dir
	return hparams


class HParams():
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

