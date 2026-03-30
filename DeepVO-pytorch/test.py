# predicted as a batch
from config.params import par
from models.model import DeepVO
from models.model_cfc_ncp import DeepVOCfC
import numpy as np
import glob
import os
import time
import torch
import argparse
from config.data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
from config.helper import eulerAnglesToRotationMatrix

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='_lstm', choices=['_lstm', '_cfc'])
	args = parser.parse_args()
	args = parser.parse_args()

	# videos_to_test = ['04', '06', '07', '10', '09']

	videos_to_test = ['00', '02', '08', '09','01', '04', '05', '06', '07', '10']
	# Path
	par.model_suffix = args.model
	par.resume_t_or_v = '.valid'  # 用验证集最好的模型
	load_model_path = par.get_load_path(par.resume_t_or_v)

   #choose the model you want to load
	save_dir = '/home/LXT/LJY/DeepVO-pytorch/results/test{}/'.format(par.model_suffix)  # e.g., 'result_cfc/' or 'result_lstm/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)


	# Load model
	if par.model_suffix == '_cfc':
		from models.model_cfc_ncp import default_ncp_config
		M_deepvo = DeepVOCfC(par.img_h, par.img_w, par.batch_norm, default_ncp_config)
	else:
		M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
	use_cuda = torch.cuda.is_available()
	if use_cuda:
		M_deepvo = M_deepvo.cuda()
		M_deepvo.load_state_dict(torch.load(load_model_path))
	else:
		M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}))
	print('Load model from: ', load_model_path)



	# Data
	n_workers = 48
	seq_len = int((par.seq_len[0]+par.seq_len[1])/2)
	overlap = seq_len - 1
	print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
	batch_size = par.batch_size


	fd=open('test_dump.txt', 'w')
	fd.write('\n'+'='*50 + '\n')


	for test_video in videos_to_test:
		df = get_data_info(folder_list=[test_video], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
		df = df.loc[df.seq_len == seq_len]  # drop last
		dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
		df.to_csv('test_df.csv')
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
		
		gt_pose = np.load('{}{}.npy'.format(par.pose_dir, test_video))  # (n_images, 6)

		# Predict
		M_deepvo.eval()
		has_predict = False
		answer = [[0.0]*6, ]
		st_t = time.time()
		n_batch = len(dataloader)

		for i, batch in enumerate(dataloader):
			print('{} / {}'.format(i, n_batch), end='\r', flush=True)
			_, x, y = batch
			if use_cuda:
				x = x.cuda()
				y = y.cuda()
			batch_predict_pose = M_deepvo.forward(x)
			if isinstance(batch_predict_pose, tuple):
				batch_predict_pose = batch_predict_pose[0]

			# Record answer
			fd.write('Batch: {}\n'.format(i))
			for seq, predict_pose_seq in enumerate(batch_predict_pose):
				for pose_idx, pose in enumerate(predict_pose_seq):
					fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))


			batch_predict_pose = batch_predict_pose.data.cpu().numpy()
			if i == 0:
				for pose in batch_predict_pose[0]:
					# use all predicted pose in the first prediction
					for i in range(len(pose)):
						# Convert predicted relative pose to absolute pose by adding last pose
						pose[i] += answer[-1][i]
					answer.append(pose.tolist())
				batch_predict_pose = batch_predict_pose[1:]

			# transform from relative to absolute 
			
			for predict_pose_seq in batch_predict_pose:
				# predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
				ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) #eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
				location = ang.dot(predict_pose_seq[-1][3:])
				predict_pose_seq[-1][3:] = location[:]

			# use only last predicted pose in the following prediction
				last_pose = predict_pose_seq[-1]
				for i in range(len(last_pose)):
					last_pose[i] += answer[-1][i]
				# normalize angle to -Pi...Pi over y axis
				last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
				answer.append(last_pose.tolist())

		print('len(answer): ', len(answer))
		print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, test_video))))
		print('Predict use {} sec'.format(time.time() - st_t))


		# Save answer
		with open('{}/out_{}.txt'.format(save_dir, test_video), 'w') as f:
			for pose in answer:
				if type(pose) == list:
					f.write(', '.join([str(p) for p in pose]))
				else:
					f.write(str(pose))
				f.write('\n')


		# Calculate loss
		gt_pose = np.load('{}{}.npy'.format(par.pose_dir, test_video))  # (n_images, 6)
		loss = 0
		for t in range(len(gt_pose)):
			angle_loss = np.sum((answer[t][:3] - gt_pose[t,:3]) ** 2)
			translation_loss = np.sum((answer[t][3:] - gt_pose[t,3:6]) ** 2)
			loss = (100 * angle_loss + translation_loss)
		loss /= len(gt_pose)
		print('Loss = ', loss)
		print('='*50)
