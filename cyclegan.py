# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import sys, time, os
sys.path.append('./utils')
from nets import *
from data import *
from scipy.misc import imsave
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from hyperboard import Agent


agent = Agent(username = '', password = '', address = '127.0.0.1', port = 5000)
d_a_loss = {'CycleGAN': 'adv loss of D_A'}
d_b_loss = {'CycleGAN': 'adv loss of D_B'}
g_ab_loss = {'CycleGAN': 'adv loss of G_AB'}
g_ba_loss = {'CycleGAN': 'adv loss of G_BA'}
a_recon_loss = {'CycleGAN': 'reconstruction loss of A (A -> B -> A)'}
b_recon_loss = {'CycleGAN': 'reconstruction loss of B (B -> A -> B)'}
da_loss = agent.register(d_a_loss, 'loss', overwrite=True)
db_loss = agent.register(d_b_loss, 'loss', overwrite=True)
g_loss_ab = agent.register(g_ab_loss, 'loss', overwrite=True)
g_loss_ba = agent.register(g_ba_loss, 'loss', overwrite=True)
g_recon_loss_a = agent.register(a_recon_loss, 'loss', overwrite=True)
g_recon_loss_b = agent.register(b_recon_loss, 'loss', overwrite=True)


# def sample_z(batch_size, z_dim):
# 	return np.random.uniform(-1., 1., size=[batch_size, z_dim])


class CycleGAN():
	def __init__(self, G_AB, G_BA, D_A, D_B, data_A, data_B):
		'''
		TODO: make it suitable for any input image size.
		Since both G are autoencoders, they can be any image size,
		the only thing we need to concern is D. Remove any linear layer 
		by using a global pooling layer and it is suitable for any
		input image size.
		'''
		self.G_AB = G_AB
		self.G_BA = G_BA
		self.D_A = D_A
		self.D_B = D_B
		self.data_A = data_A
		self.data_B = data_B
		self.cuda = True

		if self.cuda:
			self.G_AB.cuda()
			self.G_BA.cuda()
			self.D_A.cuda()
			self.D_B.cuda()

	
	def train(self, sample_dir, ckpt_dir, training_epochs=50000):
		fig_count = 0
		g_lr = 2e-3
		d_lr = 1e-3
		batch_size = 1
		lam = 1e-3
		n_g = 1

		self.data_A.reset(sort=True)
		self.data_B.reset(sort=True)

		if self.cuda:
			label_true = Variable(torch.ones(batch_size).cuda())
			label_false = Variable(torch.zeros(batch_size).cuda())
			criterion = nn.BCELoss().cuda()
		else:
			label_true = Variable(torch.ones(batch_size))
			label_false = Variable(torch.zeros(batch_size))
			criterion = nn.BCELoss()

		optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_G_AB = optim.Adam(self.G_AB.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_G_BA = optim.Adam(self.G_BA.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=1e-5)

		scheduler_G_BA = lr_scheduler.StepLR(optimizer_G_BA, step_size=5000, gamma=0.92)
		scheduler_G_AB = lr_scheduler.StepLR(optimizer_G_AB, step_size=5000, gamma=0.92)
		scheduler_D_A = lr_scheduler.StepLR(optimizer_D_A, step_size=5000, gamma=0.92)
		scheduler_D_B = lr_scheduler.StepLR(optimizer_D_B, step_size=5000, gamma=0.92)


		for epoch in range(training_epochs):
			scheduler_G_BA.step()
			scheduler_G_AB.step()
			scheduler_D_A.step()
			scheduler_D_B.step()

			begin_time = time.time()

			# update D_A and D_B
			self.D_A.zero_grad()  # clear previous gradients
			self.D_B.zero_grad()
			
			real_images_A = Variable(torch.from_numpy(self.data_A(batch_size)))
			real_images_B = Variable(torch.from_numpy(self.data_B(batch_size)))

			if self.cuda:
				real_images_A = real_images_A.cuda()
				real_images_B = real_images_B.cuda()

			d_real_A = self.D_A(real_images_A)
			d_loss_real_A = criterion(d_real_A, label_true)
			d_loss_real_A.backward()
			d_loss_A = d_loss_real_A

			fake_images_AB = self.G_AB(real_images_A)
			d_fake_AB = self.D_B(fake_images_AB.detach())
			d_loss_fake_AB = criterion(d_fake_AB, label_false)
			d_loss_fake_AB.backward()
			d_loss_B = d_loss_fake_AB

			fake_images_ABA = self.G_BA(fake_images_AB.detach())
			d_fake_ABA = self.D_A(fake_images_ABA.detach())  # detach() makes it a leaf node, so that backward will not apply to G
			d_loss_fake_ABA = criterion(d_fake_ABA, label_false)
			d_loss_fake_ABA.backward()  # gradients accmulate at D's nodes
			d_loss_A += d_loss_fake_ABA

			d_real_B = self.D_B(real_images_B)
			d_loss_real_B = criterion(d_real_B, label_true)
			d_loss_real_B.backward()
			d_loss_B += d_loss_real_B

			fake_images_BA = self.G_BA(real_images_B)
			d_fake_BA = self.D_A(fake_images_BA.detach())
			d_loss_fake_BA = criterion(d_fake_BA, label_false)
			d_loss_fake_BA.backward()
			d_loss_A += d_loss_fake_BA

			fake_images_BAB = self.G_AB(fake_images_BA.detach())
			d_fake_BAB = self.D_B(fake_images_BAB.detach())
			d_loss_fake_BAB = criterion(d_fake_BAB, label_false)
			d_loss_fake_BAB.backward()
			d_loss_B += d_loss_fake_BAB

			optimizer_D_A.step()
			optimizer_D_B.step()

			agent.append(da_loss, epoch, float(d_loss_A.data[0]))
			agent.append(db_loss, epoch, float(d_loss_B.data[0]))

			# for G_AB and G_BA
			self.G_BA.zero_grad()
			self.G_AB.zero_grad()

			d_fake_AB = self.D_B(fake_images_AB)
			g_AB_loss_fake = criterion(d_fake_AB, label_true)
			g_AB_loss_fake.backward(retain_graph=True)
			g_AB_loss = g_AB_loss_fake
			d_fake_ABA = self.D_A(fake_images_ABA)
			g_ABA_loss_fake = criterion(d_fake_ABA, label_true)
			g_ABA_loss_fake.backward(retain_graph=True)
			g_ABA_loss_recon = lam * torch.sum(torch.abs(fake_images_ABA - real_images_A)) / batch_size  # lam * torch.mean((fake_images_ABA - real_images_A)**2)
			g_ABA_loss_recon.backward(retain_graph=True)
			g_BA_loss = g_ABA_loss_recon + g_ABA_loss_fake

			d_fake_BA = self.D_A(fake_images_BA)
			g_BA_loss_fake = criterion(d_fake_BA, label_true)
			g_BA_loss_fake.backward(retain_graph=True)
			g_BA_loss += g_BA_loss_fake
			d_fake_BAB = self.D_B(fake_images_BAB)
			g_BAB_loss_fake = criterion(d_fake_BAB, label_true)
			g_BAB_loss_fake.backward(retain_graph=True)
			g_BAB_loss_recon = lam * torch.sum(torch.abs(fake_images_BAB - real_images_B)) / batch_size  # lam * torch.mean((fake_images_BAB - real_images_B)**2) 
			g_BAB_loss_recon.backward(retain_graph=True)
			g_AB_loss += (g_BAB_loss_recon + g_BAB_loss_fake)

			g_AB_recon_loss = lam * torch.mean(torch.abs(real_images_B - fake_images_AB))
			g_AB_recon_loss.backward()
			g_AB_loss += g_AB_recon_loss
			g_BA_recon_loss = lam * torch.mean(torch.abs(real_images_A - fake_images_BA))
			g_BA_recon_loss.backward()
			g_BA_loss += g_BA_recon_loss

			optimizer_G_BA.step()
			optimizer_G_AB.step()

			agent.append(g_recon_loss_a, epoch, float(g_ABA_loss_recon.data[0]+g_BA_recon_loss.data[0]))
			agent.append(g_recon_loss_b, epoch, float(g_BAB_loss_recon.data[0]+g_AB_recon_loss.data[0]))

			elapse_time = time.time() - begin_time
			print('Iter[%s], d_a_loss: %.4f, d_b_loss: %.4f, g_ab_loss: %s, g_ba_loss: %s, time elapsed: %.4fsec' % \
					(epoch+1, d_loss_A.data[0], d_loss_B.data[0], g_AB_loss.data[0], g_BA_loss.data[0], elapse_time))

			if epoch % 500 == 0:
				real_images_A = Variable(torch.from_numpy(self.data_A(batch_size)))
				if self.cuda:
					real_images_A = real_images_A.cuda()
				fake_images_AB = self.G_AB(real_images_A)
				fake_images_ABA = self.G_BA(fake_images_AB)
				A = torch.cat([real_images_A[0], fake_images_AB[0], fake_images_ABA[0]], 2)
				imsave(os.path.join(sample_dir, 'A-%s.png'%(epoch+1)), np.transpose(A.cpu().data.numpy(), [1,2,0]))
				
				real_images_B = Variable(torch.from_numpy(self.data_B(batch_size)))
				if self.cuda:
					real_images_B = real_images_B.cuda()
				fake_images_BA = self.G_BA(real_images_B)
				fake_images_BAB = self.G_AB(fake_images_BA)
				B = torch.cat([real_images_B[0], fake_images_BA[0], fake_images_BAB[0]], 2)
				imsave(os.path.join(sample_dir, 'B-%s.png'%(epoch+1)), np.transpose(B.cpu().data.numpy(), [1,2,0]))

			if epoch % 10000 == 0:
				torch.save(self.G_AB.state_dict(), os.path.join(ckpt_dir, 'G_AB_epoch-%s.pth' % epoch))
				torch.save(self.G_BA.state_dict(), os.path.join(ckpt_dir, 'G_BA_epoch-%s.pth' % epoch))
				torch.save(self.D_A.state_dict(), os.path.join(ckpt_dir, 'D_A_epoch-%s.pth' % epoch))
				torch.save(self.D_B.state_dict(), os.path.join(ckpt_dir, 'D_B_epoch-%s.pth' % epoch))

	def pretrain(self, save_dir, training_epochs=10000):
		'''
		Pretraining using pair data.
		'''
		batch_size = 1
		g_lr = 1e-5

		self.data_A.reset(sort=True)
		self.data_B.reset(sort=True)

		optimizer_G_AB = optim.SGD(self.G_AB.parameters(), lr=g_lr)
		optimizer_G_BA = optim.SGD(self.G_BA.parameters(), lr=g_lr)

		
		for epoch in range(training_epochs):
			begin_time = time.time()

			real_images_A = Variable(torch.from_numpy(self.data_A(batch_size)))
			real_images_B = Variable(torch.from_numpy(self.data_B(batch_size)))

			if self.cuda:
				real_images_A = real_images_A.cuda()
				real_images_B = real_images_B.cuda()

			fake_images_AB = self.G_AB(real_images_A)
			fake_images_ABA = self.G_BA(fake_images_AB)
			fake_images_BA = self.G_BA(real_images_B)
			fake_images_BAB = self.G_AB(fake_images_BA)

			recon_loss_A = torch.mean(torch.abs(fake_images_ABA - real_images_A))
			recon_loss_A.backward(retain_graph=True)
			recon_loss_B = torch.mean(torch.abs(fake_images_BAB - real_images_B))
			recon_loss_B.backward(retain_graph=True)
			recon_loss_AB = torch.mean(torch.abs(fake_images_AB - real_images_B))
			recon_loss_AB.backward(retain_graph=True)
			recon_loss_BA = torch.mean(torch.abs(fake_images_BA - real_images_A))
			recon_loss_BA.backward()

			optimizer_G_AB.step()
			optimizer_G_BA.step()


			elapse_time = time.time() - begin_time
			print('Iter[%s], recon loss A: %.4f, recon loss B: %.4f, recon loss AB: %.4f, recon loss BA: %.4f, time elapsed: %.4fsec' % \
					(epoch+1, recon_loss_A.data[0], recon_loss_B.data[0], recon_loss_AB.data[0], recon_loss_BA.data[0], elapse_time))

			if epoch % 500 == 0:
				real_images_A = Variable(torch.from_numpy(self.data_A(batch_size)))
				if self.cuda:
					real_images_A = real_images_A.cuda()
				fake_images_AB = self.G_AB(real_images_A)
				fake_images_ABA = self.G_BA(fake_images_AB)
				A = torch.cat([real_images_A[0], fake_images_AB[0], fake_images_ABA[0]], 2)
				imsave(os.path.join(save_dir, 'A-%s.png'%(epoch+1)), np.transpose(A.cpu().data.numpy(), [1,2,0]))
				
				real_images_B = Variable(torch.from_numpy(self.data_B(batch_size)))
				if self.cuda:
					real_images_B = real_images_B.cuda()
				fake_images_BA = self.G_BA(real_images_B)
				fake_images_BAB = self.G_AB(fake_images_BA)
				B = torch.cat([real_images_B[0], fake_images_BA[0], fake_images_BAB[0]], 2)
				imsave(os.path.join(save_dir, 'B-%s.png'%(epoch+1)), np.transpose(B.cpu().data.numpy(), [1,2,0]))


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# save generated images
	sample_dir = 'Samples/CycleGAN.l1_rc'
	ckpt_dir = 'Models/CycleGAN.l1_rc'
	pretrain_dir = 'Pretrain/CycleGAN.l1_rc'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(pretrain_dir):
		os.makedirs(pretrain_dir)

	G_AB = G_autoencoder_flexible(channel=3)
	G_BA = G_autoencoder_flexible(channel=3)
	D_A = D_conv_flexible(channel=3, activation='sigmoid')
	D_B = D_conv_flexible(channel=3, activation='sigmoid')

	print('G_AB:\n', G_AB)
	print('G_BA:\n', G_BA)
	print('D_A:\n', D_A)
	print('D_B:\n', D_B)


	data_A = image_folder(datapath='~/dataset/cat/raw', max_hw=512)
	data_B = image_folder(datapath='~/dataset/cat/pencil', max_hw=512)

	cyclegan = CycleGAN(G_AB, G_BA, D_A, D_B, data_A, data_B)
	cyclegan.pretrain(pretrain_dir, training_epochs=2000)
	cyclegan.train(sample_dir, ckpt_dir, training_epochs=500000)
