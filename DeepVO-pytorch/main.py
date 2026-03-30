import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import json
from config.params import par
from models.model import DeepVO
from config.data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset, get_partition_data_info

# Set model suffix BEFORE using any path methods
par.model_suffix = '_lstm'

# Load or initialize loss data for JSON
loss_json_path = par.get_loss_json_path()
if par.resume and os.path.exists(loss_json_path):
	with open(loss_json_path, 'r') as f:
		loss_data = json.load(f)
else:
	loss_data = {'train_loss': [], 'valid_loss': [], 'train_std': [], 'valid_std': []}

# Prepare Data
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
	print('Load data info from {}'.format(par.train_data_info_path))
	train_df = pd.read_pickle(par.train_data_info_path)
	valid_df = pd.read_pickle(par.valid_data_info_path)
else:
	print('Create new data info')
	if par.partition != None:
		partition = par.partition
		train_df, valid_df = get_partition_data_info(partition, par.train_video, par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=True)
	else:
		train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
		valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
	# save the data info
	train_df.to_pickle(par.train_data_info_path)
	valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('='*50)


# Model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo = M_deepvo.cuda()


# Load FlowNet weights pretrained with FlyingChairs
# NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
if par.pretrained_flownet and not par.resume:
	if use_cuda:
		pretrained_w = torch.load(par.pretrained_flownet)
	else:
		pretrained_w = torch.load(par.pretrained_flownet_flownet, map_location='cpu')
	print('Load FlowNet pretrained model')
	# Use only conv-layer-part of FlowNet as CNN for DeepVO
	model_dict = M_deepvo.state_dict()
	update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
	model_dict.update(update_dict)
	M_deepvo.load_state_dict(model_dict)


# Create optimizer
if par.optim['opt'] == 'Adam':
	optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
	optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
	optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'])
	T_iter = par.optim['T']*len(train_dl)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Load trained DeepVO model and optimizer
if par.resume:
	M_deepvo.load_state_dict(torch.load(par.get_load_path(par.resume_t_or_v)))
	optimizer.load_state_dict(torch.load(par.get_load_path(par.resume_t_or_v).replace('.model', '.optimizer')))
	print('Load model from: ', par.get_load_path(par.resume_t_or_v))


# Train
min_loss_t = 1e10
min_loss_v = 1e10
M_deepvo.train()
for ep in range(par.epochs):
	st_t = time.time()
	print('='*50)
	# Train
	M_deepvo.train()
	loss_mean = 0
	t_loss_list = []
	for _, t_x, t_y in train_dl:
		if use_cuda:
			t_x = t_x.cuda(non_blocking=par.pin_mem)
			t_y = t_y.cuda(non_blocking=par.pin_mem)
		ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
		t_loss_list.append(float(ls))
		loss_mean += float(ls)
		if par.optim == 'Cosine':
			lr_scheduler.step()
	print('Train take {:.1f} sec'.format(time.time()-st_t))
	loss_mean /= len(train_dl)

	# Validation
	st_t = time.time()
	M_deepvo.eval()
	loss_mean_valid = 0
	v_loss_list = []
	with torch.no_grad():
		for _, v_x, v_y in valid_dl:
			if use_cuda:
				v_x = v_x.cuda(non_blocking=par.pin_mem)
				v_y = v_y.cuda(non_blocking=par.pin_mem)
			v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
			v_loss_list.append(float(v_ls))
			loss_mean_valid += float(v_ls)
	print('Valid take {:.1f} sec'.format(time.time()-st_t))
	loss_mean_valid /= len(valid_dl)

	print('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

	# Save loss to JSON
	loss_data['train_loss'].append(float(loss_mean))
	loss_data['valid_loss'].append(float(loss_mean_valid))
	loss_data['train_std'].append(float(np.std(t_loss_list)))
	loss_data['valid_std'].append(float(np.std(v_loss_list)))
	with open(loss_json_path, 'w') as f:
		json.dump(loss_data, f, indent=2)

	# Save model
	# save if the valid loss decrease
	check_interval = 1
	if loss_mean_valid < min_loss_v and ep % check_interval == 0:
		min_loss_v = loss_mean_valid
		print('Save model at ep {}, mean of valid loss: {}'.format(ep+1, loss_mean_valid))
		torch.save(M_deepvo.state_dict(), par.get_save_path('valid'))
		torch.save(optimizer.state_dict(), par.get_save_path('valid').replace('.model', '.optimizer'))
	# save if the training loss decrease
	check_interval = 1
	if loss_mean < min_loss_t and ep % check_interval == 0:
		min_loss_t = loss_mean
		print('Save model at ep {}, mean of train loss: {}'.format(ep+1, loss_mean))
		torch.save(M_deepvo.state_dict(), par.get_save_path('train'))
		torch.save(optimizer.state_dict(), par.get_save_path('train').replace('.model', '.optimizer'))
