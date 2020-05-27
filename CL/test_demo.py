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
args = parser.parse_args()
model = {
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE
}
con = config.Config()
con.set_max_epoch(15)
con.load_test_data()
con.set_test_model(model[args.model_name])
con.set_epoch_range([7,12])
con.test()
