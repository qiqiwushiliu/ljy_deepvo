import os

class Parameters():
	def __init__(self):
		self.n_processors = 4
		# Path
		self.data_dir =  '/data3/LXT/LJY/KITTI_20241217_142029/images'
		self.image_dir = '/data3/LXT/LJY/KITTI_20241217_142029/images/'
		self.pose_dir = '/home/LXT/LJY/DeepVO-pytorch/poses/'

		self.train_video = ['00', '01', '02', '05', '08', '09']
		self.valid_video = ['04', '06', '07', '10']
		self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8

		# Data Preprocessing
		self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
		self.img_w = 1226   # original size is about 1226
		self.img_h = 370  # original size is about 370
		self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
		self.img_stds =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
		self.minus_point_5 = True

		self.seq_len = (5, 7)
		self.sample_times = 3

		# Model suffix to distinguish different runs (e.g., '_cfc', '_lstm')
		# Set this BEFORE using get_model_path() or get_save_path()
		self.model_suffix = ''

		# Data info path
		self.train_data_info_path = '/home/LXT/LJY/DeepVO-pytorch/results/datainfo/train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)
		self.valid_data_info_path = '/home/LXT/LJY/DeepVO-pytorch/results/datainfo/valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(''.join(self.train_video), ''.join(self.valid_video), self.partition, self.seq_len[0], self.seq_len[1], self.sample_times)

		# Model
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0   # 0: no dropout
		self.clip = None
		self.batch_norm = True
		# Training
		self.epochs = 250
		self.batch_size = 8
		self.pin_mem = False
		self.optim = {'opt': 'Adam', 'lr': 0.001}
					# Choice:
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

		# Pretrain, Resume training
		self.pretrained_flownet = '/home/LXT/LJY/DeepVO-pytorch/pretrained/flownets_bn_EPE2.459.pth'
								# Choice:
								# None
								# './pretrained/flownets_bn_EPE2.459.pth.tar'
								# './pretrained/flownets_EPE1.951.pth.tar'
		self.resume = False # resume training
		self.resume_t_or_v = '.train'

		# Records directory
		self.record_path = '/home/LXT/LJY/DeepVO-pytorch/results/records'
		self.save_model_path = '/home/LXT/LJY/DeepVO-pytorch/weights'

		# Create directories if they don't exist
		if not os.path.isdir(self.record_path):
			os.makedirs(self.record_path, exist_ok=True)
		if not os.path.isdir(self.save_model_path):
			os.makedirs(self.save_model_path, exist_ok=True)
		if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
			os.makedirs(os.path.dirname(self.train_data_info_path), exist_ok=True)

	def get_model_name(self):
		"""Generate model file name based on current settings."""
		return 't{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}{}'.format(
			''.join(self.train_video),
			''.join(self.valid_video),
			self.img_h, self.img_w,
			self.seq_len[0], self.seq_len[1],
			self.batch_size,
			self.rnn_hidden_size,
			'_'.join([k+str(v) for k, v in self.optim.items()]),
			self.model_suffix
		)

	def get_load_path(self, suffix='.train'):
		"""Get full path for loading a model checkpoint."""
		return os.path.join(self.save_model_path, self.get_model_name() + '.model' + suffix)

	def get_save_path(self, which='valid'):
		"""Get full path for saving a model checkpoint (which='valid' or 'train')."""
		return os.path.join(self.save_model_path, self.get_model_name() + '.model.' + which)

	def get_record_path(self):
		"""Get full path for the training record file."""
		return os.path.join(self.record_path, self.get_model_name() + '.txt')

	def get_loss_json_path(self):
		"""Get full path for the loss JSON file."""
		return os.path.join(self.record_path, self.get_model_name() + '_loss.json')

par = Parameters()
