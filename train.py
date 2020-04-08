import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'cnn_att', help = 'name of the model')
parser.add_argument('--dataset', type = str, default = 'Baike', help = 'name of the dataset')
parser.add_argument('--learning_rate', type=float, default=0.5, help='the learning rate')
parser.add_argument('--sen_hidden_size', type=int, default=150, help='hidden size of sentence')
parser.add_argument('--path_hidden_size', type=int, default=150, help='hidden size of path')
parser.add_argument('--batch_size', type=int, default=80, help='batch size')
args = parser.parse_args()
model = {
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE
}
con = config.Config(args)
con.set_max_epoch(60)
con.load_train_data()
con.load_test_data()
con.set_train_model(model[args.model_name])
con.train()
