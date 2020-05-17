import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class _CNN_S(nn.Module):
	def __init__(self, config):
		super(_CNN_S, self).__init__()
		self.config = config
		self.in_channels = 1
		self.in_height = self.config.max_length
		self.in_width = self.config.word_size + 2 * self.config.pos_size +self.config.tag_size
		self.kernel_size = (self.config.window_size, self.in_width)
		self.out_channels = self.config.hidden_size
		self.stride = (1, 1)
		self.padding = (1, 0)
		self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
	def forward(self, embedding):
		return self.cnn(embedding)

class _CNN_P(nn.Module):
	def __init__(self, config):
		super(_CNN_P, self).__init__()
		self.config = config
		self.in_channels = 1
		self.in_height = self.config.max_length
		self.in_width = self.config.word_size +self.config.tag_size + self.config.deprel_vec.shape[1]+self.config.dir_size
		self.kernel_size = (self.config.window_size, self.in_width)
		self.out_channels = self.config.hidden_size
		self.stride = (1, 1)
		self.padding = (1, 0)
		self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
	def forward(self, embedding):
		return self.cnn(embedding)

class _PiecewisePooling(nn.Module):
	def __init(self):
		super(_PiecewisePooling, self).__init__()
	def forward(self, x, mask, hidden_size):
		mask = torch.unsqueeze(mask, 1)
		# print((mask+x).size())
		x, _ = torch.max(mask + x, dim = 2)
		x = x - 100
		return x.view(-1, hidden_size * 3)

class _MaxPooling(nn.Module):
	def __init__(self):
		super(_MaxPooling, self).__init__()
	def forward(self, x, hidden_size):
		x, _ = torch.max(x, dim = 2)
		return x.view(-1, hidden_size)

class PCNN(nn.Module):
	def __init__(self, config):
		super(PCNN, self).__init__()
		self.config = config
		self.mask = None
		self.cnn = _CNN_S(config)
		self.pooling = _PiecewisePooling()
		self.activation = nn.ReLU()
	def forward(self, embedding):
		embedding = torch.unsqueeze(embedding, dim = 1)
		x = self.cnn(embedding)
		x = self.pooling(x, self.mask, self.config.hidden_size)
		return self.activation(x)

class CNN(nn.Module):
	def __init__(self, config):
		super(CNN, self).__init__()
		self.config = config
		self.cnn_s = _CNN_S(config)
		self.cnn_p = _CNN_P(config)
		self.pooling = _MaxPooling()
		self.activation = nn.ReLU()
	def forward(self, embed):
		sen_embedding = embed[0]
		MDP_embedding = embed[1]
		head = embed[2]
		tail = embed[3]
		sen_embedding = torch.unsqueeze(sen_embedding, dim = 1)
		MDP_embedding = torch.unsqueeze(MDP_embedding, dim=1)
		x = self.cnn_s(sen_embedding)
		y = self.cnn_p(MDP_embedding)
		x = self.pooling(x, self.config.hidden_size)
		y = self.pooling(y, self.config.hidden_size)
		# x = self.activation(torch.cat((x,y), dim=1))

		# x = torch.cat([x, y, head, tail], 1)
		return x

class _GRU(nn.Module):
	def __init__(self, config):
		super(_GRU, self).__init__()
		self.config = config
		# self.in_channels = 1
		# self.in_height = self.config.max_length
		self.in_width = self.config.word_size + 2 * self.config.pos_size
		self.out_channels = self.config.hidden_size
		self.gru = nn.GRU(self.in_width, self.out_channels, bidirectional=True)

	def forward(self, embedding):
		return self.gru(embedding)

class GRU(nn.Module):
	def __init__(self, config):
		super(GRU, self).__init__()
		self.config = config
		self.gru = _GRU(config)
		self.att = nn.Linear(self.config.hidden_size, 1)
		self.activation = nn.Tanh()
	def forward(self, embedding):
		x, _ = self.gru(embedding)
		x_f = x[:, :, :self.config.hidden_size]
		x_b = x[:, :, self.config.hidden_size:]
		x = x_f + x_b
		M = self.activation(x)
		alpha = F.softmax(self.att(M), 1)
		x = torch.transpose(x, 1, 2)
		H = torch.matmul(x, alpha).squeeze(2)
		return self.activation(H)