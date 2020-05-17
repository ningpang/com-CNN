import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Embedding(nn.Module):
	def __init__(self, config):
		super(Embedding, self).__init__()
		self.config = config
		self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
		self.pos1_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx = 0)
		self.pos2_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx = 0)
		self.tag_embedding = nn.Embedding(self.config.tag_num, self.config.tag_size, padding_idx=0)
		self.dir_embedding = nn.Embedding(self.config.dir_num, self.config.dir_size, padding_idx=0)
		self.deprel_embedding = nn.Embedding(self.config.deprel_vec.shape[0], self.config.deprel_vec.shape[1], padding_idx=3)
		self.project_dim = self.config.data_word_vec.shape[1]+2*self.config.pos_size+self.config.tag_size

		self.init_word_weights()
		self.init_pos_weights()
		self.init_tag_weights()
		self.init_derel_weights()
		self.project_sen = nn.Linear(self.project_dim, self.config.data_word_vec.shape[1], bias=True)
		self.project_ent = nn.Linear(self.config.data_word_vec.shape[1], self.config.data_word_vec.shape[1], bias=True)
		self.attention = nn.Linear(2*self.config.data_word_vec.shape[1], 1)
		self.bias_act = nn.Parameter(torch.Tensor(2*self.config.data_word_vec.shape[1]))
		self.bias_att = nn.Parameter(torch.Tensor(2*self.config.data_word_vec.shape[1]))
		self.activiation = nn.Tanh()

		self.word = None
		self.pos1 = None
		self.pos2 = None
		self.head = None
		self.tail = None
		self.root = None
		self.tag = None
		self.MDPword = None
		self.MDPpos = None
		self.MDPrel = None
		self.MDPdir = None
	def init_word_weights(self):
		self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))
	def init_pos_weights(self):
		nn.init.xavier_uniform(self.pos1_embedding.weight.data)
		if self.pos1_embedding.padding_idx is not None:
			self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
		nn.init.xavier_uniform(self.pos2_embedding.weight.data)
		if self.pos2_embedding.padding_idx is not None:
			self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)
	def init_tag_weights(self):
		nn.init.xavier_uniform(self.tag_embedding.weight.data)
		if self.tag_embedding.padding_idx is not None:
			self.tag_embedding.weight.data[self.tag_embedding.padding_idx].fill_(0)
	def init_dir_weights(self):
		nn.init.xavier_uniform(self.tag_embedding.weight.data)
		if self.tag_embedding.padding_idx is not None:
			self.tag_embedding.weight.data[self.tag_embedding.padding_idx].fill_(0)
	# def init_deprel_weights(self):
	# 	self.deprel_embedding.weight.data.copy_(torch.from_numpy(self.config.deprel_vec))
	def init_derel_weights(self):
		nn.init.xavier_uniform(self.deprel_embedding.weight.data)
		if self.deprel_embedding.padding_idx is not None:
			self.deprel_embedding.weight.data[self.deprel_embedding.padding_idx].fill_(0)
	def forward(self):
		word = self.word_embedding(self.word)
		pos1 = self.pos1_embedding(self.pos1)
		pos2 = self.pos2_embedding(self.pos2)
		tag = self.tag_embedding(self.tag)

		MDPword = self.word_embedding(self.MDPword)
		MDPpos = self.tag_embedding(self.MDPpos)
		MDPrel = self.deprel_embedding(self.MDPrel)
		MDPdir = self.dir_embedding(self.MDPdir)

		head = self.word_embedding(self.head)
		tail = self.word_embedding(self.tail)
		root = self.word_embedding(self.root)
		sen_embedding = torch.cat((word, pos1, pos2, tag), dim = 2) # n*length*dim
		MDP_embedding = torch.cat((MDPword, MDPpos, MDPrel, MDPdir), dim=2)
		feature = head - tail
		# sen_embedding = self.entity_attention(sen_embedding, feature)
		# print(feature.size())
		# print(ll)
		return sen_embedding, MDP_embedding, head, tail

	def entity_attention(self, sentence, entity):
		ent = self.project_ent(entity.unsqueeze(1).expand(-1, self.config.max_length, -1))
		sent = self.project_sen(sentence)
		att_score = (ent*sent).sum(2).unsqueeze(2)
		sentence = sentence*att_score
		return sentence

