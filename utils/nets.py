import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
try:
	reduce
except Exception as e:
	from functools import reduce


def compute_conv_output_and_padding_param(in_h, in_w, stride, kernel_size):
	out_h, out_w = (in_h + stride - 1) // stride, (in_w + stride - 1) // stride
	padding_h = (stride * out_h - stride + kernel_size - in_h + 1) // 2
	padding_w = (stride * out_w - stride + kernel_size - in_w + 1) // 2
	return (padding_h, padding_w), (out_h, out_w)

def compute_deconv_padding_param(in_h, in_w, out_h, out_w, stride, kernel_size):
	padding_h = padding_w = kernel_size // 2
	output_padding_h = out_h - (in_h - 1) * stride + 2 * padding_h - kernel_size
	output_padding_w = out_w - (in_w - 1) * stride + 2 * padding_w - kernel_size
	return (padding_h, padding_w), (output_padding_h, output_padding_w)


class D_autoencoder_flexible(nn.Module):
	def __init__(self, channel=3, stride=2, kernel_size=3, activation='sigmoid'):
		super(D_autoencoder_flexible, self).__init__()
		self.channel = channel
		self.stride = stride
		self.kernel_size = kernel_size
		self.activation = activation

		# activation
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		self.relu = nn.ReLU()
		if self.activation.lower() == 'sigmoid':
			self.act_fn = nn.Sigmoid()
		elif self.activation.lower() == 'tanh':
			self.act_fn = nn.Tanh()
		else:
			raise ValueError('Unsupported activation type: %s' % self.activation)


		# weights
		self.conv1_weight = Parameter(torch.normal(std=0.02*torch.ones(64, 3, self.kernel_size, self.kernel_size)))
		self.conv2_weight = Parameter(torch.normal(std=0.02*torch.ones(128, 64, self.kernel_size, self.kernel_size)))
		self.conv3_weight = Parameter(torch.normal(std=0.02*torch.ones(256, 128, self.kernel_size, self.kernel_size)))
		self.conv4_weight = Parameter(torch.normal(std=0.02*torch.ones(512, 256, self.kernel_size, self.kernel_size)))
		self.conv_trans1_weight = Parameter(torch.normal(std=0.02*torch.ones(512, 256, self.kernel_size, self.kernel_size)))
		self.conv_trans2_weight = Parameter(torch.normal(std=0.02*torch.ones(256, 128, self.kernel_size, self.kernel_size)))
		self.conv_trans3_weight = Parameter(torch.normal(std=0.02*torch.ones(128, 64, self.kernel_size, self.kernel_size)))
		self.conv_trans4_weight = Parameter(torch.normal(std=0.02*torch.ones(64, 3, self.kernel_size, self.kernel_size)))
		# self.register_parameter('conv1.weight', self.conv1_weight)
		# self.register_parameter('conv2.weight', self.conv2_weight)
		# self.register_parameter('conv3.weight', self.conv3_weight)
		# self.register_parameter('conv4.weight', self.conv4_weight)
		# self.register_parameter('conv_trans1.weight', self.conv_trans1_weight)
		# self.register_parameter('conv_trans2.weight', self.conv_trans2_weight)
		# self.register_parameter('conv_trans3.weight', self.conv_trans3_weight)
		# self.register_parameter('conv_trans4.weight', self.conv_trans4_weight)


		# batch norm
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(512)
		self.bn5 = nn.BatchNorm2d(256)
		self.bn6 = nn.BatchNorm2d(128)
		self.bn7 = nn.BatchNorm2d(64)

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)


	def forward(self, x):
		padding1, out1 = compute_conv_output_and_padding_param(x.size()[2], x.size()[3], self.stride, self.kernel_size)
		d = F.conv2d(x, self.conv1_weight, stride=self.stride, padding=padding1)
		d = self.lrelu(self.bn1(d))
		# print(d.size())

		padding2, out2 = compute_conv_output_and_padding_param(out1[0], out1[1], self.stride, self.kernel_size)
		d = F.conv2d(d, self.conv2_weight, stride=self.stride, padding=padding2)
		d = self.lrelu(self.bn2(d))
		# print(d.size())

		padding3, out3 = compute_conv_output_and_padding_param(out2[0], out2[1], self.stride, self.kernel_size)
		d = F.conv2d(d, self.conv3_weight, stride=self.stride, padding=padding3)
		d = self.lrelu(self.bn3(d))
		# print(d.size())

		padding4, out4 = compute_conv_output_and_padding_param(out3[0], out3[1], self.stride, self.kernel_size)
		d = F.conv2d(d, self.conv4_weight, stride=self.stride, padding=padding4)
		d = self.lrelu(self.bn4(d))
		# print(d.size())

		padding5, output_padding5 = compute_deconv_padding_param(out4[0], out4[1], out3[0], out3[1], self.stride, self.kernel_size)
		d = F.conv_transpose2d(d, self.conv_trans1_weight, stride=self.stride, padding=padding5, output_padding=output_padding5)
		d = self.relu(self.bn5(d))
		# print(d.size())

		padding6, output_padding6 = compute_deconv_padding_param(out3[0], out3[1], out2[0], out2[1], self.stride, self.kernel_size)
		d = F.conv_transpose2d(d, self.conv_trans2_weight, stride=self.stride, padding=padding6, output_padding=output_padding6)
		d = self.relu(self.bn6(d))
		# print(d.size())

		padding7, output_padding7 = compute_deconv_padding_param(out2[0], out2[1], out1[0], out1[1], self.stride, self.kernel_size)
		d = F.conv_transpose2d(d, self.conv_trans3_weight, stride=self.stride, padding=padding7, output_padding=output_padding7)
		d = self.relu(self.bn7(d))
		# print(d.size())

		padding8, output_padding8 = compute_deconv_padding_param(out1[0], out1[1], x.size()[2], x.size()[3], self.stride, self.kernel_size)
		d = F.conv_transpose2d(d, self.conv_trans4_weight, stride=self.stride, padding=padding8, output_padding=output_padding8)
		d = self.act_fn(d)
		# print(d.size())

		return d


G_autoencoder_flexible = D_autoencoder_flexible


class D_conv_flexible(nn.Module):
	def __init__(self, channel=3, last_layer_with_activation=True, stride=2, kernel_size=3, activation='sigmoid'):
		super(D_conv_flexible, self).__init__()
		self.channel = channel
		self.last_layer_with_activation = last_layer_with_activation
		self.stride = stride
		self.kernel_size = kernel_size
		self.activation = activation


		# weights
		self.conv1_weight = Parameter(torch.normal(std=0.02*torch.ones(64, 3, self.kernel_size, self.kernel_size)))
		self.conv2_weight = Parameter(torch.normal(std=0.02*torch.ones(128, 64, self.kernel_size, self.kernel_size)))
		self.conv3_weight = Parameter(torch.normal(std=0.02*torch.ones(256, 128, self.kernel_size, self.kernel_size)))
		self.conv4_weight = Parameter(torch.normal(std=0.02*torch.ones(512, 256, self.kernel_size, self.kernel_size)))
		self.linear_weight = Parameter(torch.normal(std=0.02*torch.ones(1, 512)))

		# batch norm
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(256)
		self.bn4 = nn.BatchNorm2d(512)
		# self.bn5 = nn.BatchNorm2d(1)

		# activation
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		if self.activation.lower() == 'sigmoid':
			self.act_fn = nn.Sigmoid()
		elif self.activation.lower() == 'tanh':
			self.act_fn = nn.Tanh()
		else:
			raise ValueError('Unsupported activation type: %s' % self.activation)

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.weight.data.normal_(1.0, 0.02)
				m.bias.data.fill_(0)

	def forward(self, x):
		padding1, out1 = compute_conv_output_and_padding_param(x.size()[2], x.size()[3], self.stride, self.kernel_size)
		d = F.conv2d(x, self.conv1_weight, stride=self.stride, padding=padding1)
		d = self.lrelu(self.bn1(d))
		# print(d.size())

		padding2, out2 = compute_conv_output_and_padding_param(out1[0], out1[1], self.stride, self.kernel_size)
		d = F.conv2d(d, self.conv2_weight, stride=self.stride, padding=padding2)
		d = self.lrelu(self.bn2(d))
		# print(d.size())

		padding3, out3 = compute_conv_output_and_padding_param(out2[0], out2[1], self.stride, self.kernel_size)
		d = F.conv2d(d, self.conv3_weight, stride=self.stride, padding=padding3)
		d = self.lrelu(self.bn3(d))
		# print(d.size())

		padding4, out4 = compute_conv_output_and_padding_param(out3[0], out3[1], self.stride, self.kernel_size)
		d = F.conv2d(d, self.conv4_weight, stride=self.stride, padding=padding4)
		d = self.lrelu(self.bn4(d))
		# print(d.size())

		d = F.avg_pool2d(d, d.size()[2:])
		d = d.view(d.size()[0], -1)
		d = F.linear(d, self.linear_weight)

		if self.last_layer_with_activation:
			d = self.act_fn(d)
		return d
