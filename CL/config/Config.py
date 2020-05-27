#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm

def to_var(x):
	return Variable(torch.from_numpy(x).cuda())

class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self, args):
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './data'
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.max_length = 80
		self.MDP_length = 5
		self.pos_num = 2 * self.max_length
		self.tag_num = 30
		self.dir_num = 4
		self.num_classes = 7
		self.sen_hidden_size = 150
		self.path_hidden_size = 30
		self.pos_size = 5
		self.tag_size = 10
		self.dir_size = 5
		self.max_epoch = 60
		self.opt_method = 'SGD'
		self.optimizer = None
		self.learning_rate = 0.5
		self.weight_decay = 1e-5
		self.drop_prob = 0.5
		self.checkpoint_dir = './checkpoint'
		self.test_result_dir = './test_result'
		self.save_epoch = 1
		self.test_epoch = 1
		self.pretrain_model = None
		self.trainModel = None
		self.testModel = None
		self.batch_size = 160
		self.word_size = 50
		self.dataset = args.dataset
		self.window_size = 5
		self.epoch_range = None

	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_MDP_length(self, MDP_length):
		self.MDP_length = MDP_length
		self.pos_num = 2 * self.max_length
	def set_dir_num(self, dir_num):
		self.dir_num = dir_num
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_hidden_size(self, hidden_size):
		self.hidden_size = hidden_size
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_pos_size(self, pos_size):
		self.pos_size = pos_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_dir_size(self, dir_size):
		self.dir_size = dir_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_save_epoch(self, save_epoch):
		self.save_epoch = save_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range
	
	def load_train_data(self):
		print("Reading training data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.deprel_vec = np.load(os.path.join(self.data_path, 'deprel_vec.npy'))
		self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))
		self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))
		self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))
		self.data_train_tag = np.load(os.path.join(self.data_path, 'train_tag.npy'))
		self.data_train_MDPword = np.load(os.path.join(self.data_path, 'train_MDPword.npy'))
		self.data_train_MDPrel = np.load(os.path.join(self.data_path, 'train_MDPrel.npy'))
		self.data_train_MDPpos = np.load(os.path.join(self.data_path, 'train_MDPtag.npy'))
		self.data_train_MDPdir = np.load(os.path.join(self.data_path, 'train_MDPdir.npy'))
		self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
		self.data_train_head = np.load(os.path.join(self.data_path, 'train_head.npy'))
		self.data_train_tail = np.load(os.path.join(self.data_path, 'train_tail.npy'))
		self.data_train_root = np.load(os.path.join(self.data_path, 'train_root.npy'))
		# self.data_train_ent_pair = json.load(open(os.path.join(self.data_path, 'train_bag_pair.json')))
		if self.use_bag:
			self.data_train_ent_pair = json.load(open(os.path.join(self.data_path, 'train_bag_pair.json')))
			self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))
		else:
			self.data_train_ent_pair = json.load(open(os.path.join(self.data_path, 'train_ent_pair.json')))
			self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
			self.data_train_scope = np.load(os.path.join(self.data_path, 'train_ins_scope.npy'))
		print("Finish reading")
		self.train_order = list(range(len(self.data_train_label)))
		self.train_batches = int(len(self.data_train_label) / self.batch_size)
		if len(self.data_train_label) % self.batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		print("Reading testing data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
		self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
		self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
		self.data_test_tag = np.load(os.path.join(self.data_path, 'test_tag.npy'))
		self.data_test_MDPword = np.load(os.path.join(self.data_path, 'test_MDPword.npy'))
		self.data_test_MDPrel = np.load(os.path.join(self.data_path, 'test_MDPrel.npy'))
		self.data_test_MDPpos = np.load(os.path.join(self.data_path, 'test_MDPtag.npy'))
		self.data_test_MDPdir = np.load(os.path.join(self.data_path, 'test_MDPdir.npy'))
		self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
		self.data_test_head = np.load(os.path.join(self.data_path, 'test_head.npy'))
		self.data_test_tail = np.load(os.path.join(self.data_path, 'test_tail.npy'))
		self.data_test_root = np.load(os.path.join(self.data_path, 'test_root.npy'))

		if self.use_bag:
			self.data_test_ent_pair = json.load(open(os.path.join(self.data_path, 'test_bag_pair.json')))
			self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
		else:
			self.data_test_ent_pair = json.load(open(os.path.join(self.data_path, 'test_ent_pair.json')))
			self.data_test_label = np.load(os.path.join(self.data_path, 'test_ins_label.npy'))
			self.data_test_scope = np.load(os.path.join(self.data_path, 'test_ins_scope.npy'))
		print("Finish reading")
		self.test_batches = int(len(self.data_test_label) / self.batch_size)
		if len(self.data_test_label) % self.batch_size != 0:
			self.test_batches += 1

		self.total_recall = self.data_test_label[:, 0:6].sum()

	def set_train_model(self, model):
		print("Initializing training model...")
		self.model = model
		self.trainModel = self.model(config = self)
		if self.pretrain_model != None:
			self.trainModel.load_state_dict(torch.load(self.pretrain_model))
		self.trainModel.cuda()
		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.learning_rate, lr_decay = self.lr_decay, weight_decay = self.weight_decay)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		else:
			self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
		print("Finish initializing")
 			 
	def set_test_model(self, model):
		print("Initializing test model...")
		self.model = model
		self.testModel = self.model(config = self)
		self.testModel.cuda()
		self.testModel.eval()
		print("Finish initializing")

	def get_train_batch(self, batch):
		input_scope = np.take(self.data_train_scope, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		self.batch_word = self.data_train_word[index, :]
		self.batch_tag = self.data_train_tag[index, :]
		self.batch_pos1 = self.data_train_pos1[index, :]
		self.batch_pos2 = self.data_train_pos2[index, :]
		self.batch_MDPword = self.data_train_MDPword[index, :]
		self.batch_MDPrel = self.data_train_MDPrel[index, :]
		self.batch_MDPpos = self.data_train_MDPpos[index, :]
		self.batch_MDPdir = self.data_train_MDPdir[index, :]
		self.batch_mask = self.data_train_mask[index, :]
		self.batch_head = self.data_train_head[index]
		self.batch_tail = self.data_train_tail[index]
		self.batch_root = self.data_train_root[index]
		self.batch_label = np.take(self.data_train_label, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
		self.batch_attention_query = self.data_query_label[index]
		self.batch_scope = scope
	
	def get_test_batch(self, batch):
		input_scope = self.data_test_scope[batch * self.batch_size : (batch + 1) * self.batch_size]
		index = []
		scope = [0]
		for num in input_scope:
			index = index + list(range(num[0], num[1] + 1))
			scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
		self.batch_word = self.data_test_word[index, :]
		self.batch_pos1 = self.data_test_pos1[index, :]
		self.batch_pos2 = self.data_test_pos2[index, :]
		self.batch_tag = self.data_test_tag[index, :]
		self.batch_MDPword = self.data_test_MDPword[index, :]
		self.batch_MDPrel = self.data_test_MDPrel[index, :]
		self.batch_MDPpos = self.data_test_MDPpos[index, :]
		self.batch_MDPdir = self.data_test_MDPdir[index, :]
		self.batch_mask = self.data_test_mask[index, :]
		self.batch_head = self.data_test_head[index]
		self.batch_tail = self.data_test_tail[index]
		self.batch_root = self.data_test_root[index]
		self.batch_scope = scope
	def train_one_step(self):
		self.trainModel.embedding.word = to_var(self.batch_word)
		self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
		self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
		self.trainModel.embedding.tag = to_var(self.batch_tag)
		self.trainModel.embedding.MDPword = to_var(self.batch_MDPword)
		self.trainModel.embedding.MDPrel = to_var(self.batch_MDPrel)
		self.trainModel.embedding.MDPpos = to_var(self.batch_MDPpos)
		self.trainModel.embedding.MDPdir = to_var(self.batch_MDPdir)
		self.trainModel.embedding.head = to_var(self.batch_head)
		self.trainModel.embedding.tail = to_var(self.batch_tail)
		self.trainModel.embedding.root = to_var(self.batch_root)
		self.trainModel.encoder.mask = to_var(self.batch_mask)
		self.trainModel.selector.scope = self.batch_scope
		self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
		self.trainModel.selector.label = to_var(self.batch_label)
		self.trainModel.classifier.label = to_var(self.batch_label)
		self.optimizer.zero_grad()
		loss, _output = self.trainModel()
		loss.backward()
		self.optimizer.step()
		for i, prediction in enumerate(_output):
			if self.batch_label[i] == 0:
				self.acc_NA.add(prediction == self.batch_label[i])
			else:
				self.acc_not_NA.add(prediction == self.batch_label[i])
			self.acc_total.add(prediction == self.batch_label[i])
		return loss.data[0]

	def test_one_step(self):
		self.testModel.embedding.word = to_var(self.batch_word)
		self.testModel.embedding.pos1 = to_var(self.batch_pos1)
		self.testModel.embedding.pos2 = to_var(self.batch_pos2)
		self.testModel.embedding.tag = to_var(self.batch_tag)
		self.testModel.embedding.MDPword = to_var(self.batch_MDPword)
		self.testModel.embedding.MDPrel = to_var(self.batch_MDPrel)
		self.testModel.embedding.MDPpos = to_var(self.batch_MDPpos)
		self.testModel.embedding.MDPdir = to_var(self.batch_MDPdir)
		self.testModel.embedding.head = to_var(self.batch_head)
		self.testModel.embedding.tail = to_var(self.batch_tail)
		self.testModel.embedding.root = to_var(self.batch_root)
		self.testModel.encoder.mask = to_var(self.batch_mask)
		self.testModel.selector.scope = self.batch_scope
		return self.testModel.test()

	def train(self):
		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)
		best_auc = 0.0
		best_p = None
		best_r = None
		test_triple = None
		best_epoch = 0
		for epoch in range(self.max_epoch):
			print('Epoch ' + str(epoch) + ' starts...')
			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()
			np.random.shuffle(self.train_order)
			# print(self.train_batches)
			for batch in range(self.train_batches):
				self.get_train_batch(batch)
				loss = self.train_one_step()
				time_str = datetime.datetime.now().isoformat()
				sys.stdout.write("epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))	
				sys.stdout.flush()
			if (epoch + 1) % self.save_epoch == 0:
				print('Epoch ' + str(epoch) + ' has finished')
				print('Saving model...')
				path = os.path.join(self.checkpoint_dir, self.dataset+'-'+self.model.__name__ + '-' + str(epoch))
				torch.save(self.trainModel.state_dict(), path)
				print('Have saved model to ' + path)
			if (epoch + 1) % self.test_epoch == 0 and epoch > 20:
				self.testModel = self.trainModel
				auc, pr_x, pr_y, triple = self.test_one_epoch()
				if auc > best_auc:
					best_auc = auc
					best_p = pr_x
					best_r = pr_y
					best_epoch = epoch
					test_triple = triple
		print("Finish training")
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		log = open(self.dataset+'-'+'finallog.txt', 'a')
		log.write(self.dataset+'-'+self.model.__name__+': Best epoch = '+ str(best_epoch)+ ' | auc = '+str(best_auc)+'\n')
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
		json.dump(test_triple, open(os.path.join(self.test_result_dir, self.model.__name__ + 'test_triple.json'), 'w'))
		print("Finish storing")
	def test_one_epoch(self):
		test_score = []
		for batch in tqdm(range(self.test_batches)):
			self.get_test_batch(batch)
			batch_score = self.test_one_step()
			test_score = test_score + batch_score
		test_result = []
		test_triple = []
		# for i in range(len(test_score)):
		# 	for j in range(1, len(test_score[i])):
		# 		test_result.append([self.data_test_label[i][j], test_score[i][j]])
		for i in range(len(test_score)):
			score = test_score[i][0]
			rel = 0
			for j in range(1, len(test_score[i])):
				if test_score[i][j] >= score:
					score = test_score[i][j]
					rel = j
			if rel < 6:
				test_result.append([self.data_test_label[i][rel], test_score[i][rel]])
			# test_triple.append({'entity_pair': self.data_test_ent_pair[i], 'flag': str(self.data_test_label[i][rel]),
			# 					'relation': str(rel), 'score': str(test_score[i][rel])})
		test_result = sorted(test_result, key = lambda x: x[1])
		test_result = test_result[::-1]
		pr_x = []
		pr_y = []
		correct = 0
		for i, item in enumerate(test_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / self.total_recall)
		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		print("auc: ", auc)
		return auc, pr_x, pr_y, test_triple
	def test(self):
		best_epoch = None
		best_auc = 0.0
		best_p = None
		best_r = None
		best_triple = None
		for epoch in self.epoch_range:
			path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
			if not os.path.exists(path):
				continue
			print("Start testing epoch %d" % (epoch))
			self.testModel.load_state_dict(torch.load(path))
			auc, p, r, triple = self.test_one_epoch()
			if auc > best_auc:
				best_auc = auc
				best_epoch = epoch
				best_p = p
				best_r = r
				best_triple = triple
			print("Finish testing epoch %d" % (epoch))
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		if not os.path.isdir(self.test_result_dir):
			os.mkdir(self.test_result_dir)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
		np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
		json.dump(best_triple, open(os.path.join(self.test_result_dir, self.model.__name__ + 'test_triple.json'), 'w'))
		print("Finish storing")
