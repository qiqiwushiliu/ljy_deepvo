import os
import glob
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import time
from config.params import par
from config.helper import normalize_angle_delta


def get_data_info(folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
        fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, folder))
        fpaths.sort()
        # Fixed seq_len
        if seq_len_range[0] == seq_len_range[1]:
            if sample_times > 1:
                sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
                start_frames = list(range(0, seq_len_range[0], sample_interval))
                print('Sample start from frame {}'.format(start_frames))
            else:
                start_frames = [0]

            for st in start_frames:
                seq_len = seq_len_range[0]
                n_frames = len(fpaths) - st
                jump = seq_len - overlap
                res = n_frames % seq_len
                if res != 0:
                    n_frames = n_frames - res
                x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
                y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
                Y += y_segs
                X_path += x_segs
                X_len += [len(xs) for xs in x_segs]
        # Random segment to sequences with diff lengths
        else:
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(sample_times):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path.append(x_seg)
                        if not pad_y:
                            Y.append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 15))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y.append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df


def get_partition_data_info(partition, folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    X_path = [[], []]
    Y = [[], []]
    X_len = [[], []]
    df_list = []

    for part in range(2):
        for folder in folder_list:
            start_t = time.time()
            poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
            fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, folder))
            fpaths.sort()


            # Get the middle section as validation set
            n_val = int((1-partition)*len(fpaths))
            st_val = int((len(fpaths)-n_val)/2)
            ed_val = st_val + n_val
            print('st_val: {}, ed_val:{}'.format(st_val, ed_val))
            if part == 1:
                fpaths = fpaths[st_val:ed_val]
                poses = poses[st_val:ed_val]
            else:
                fpaths = fpaths[:st_val] + fpaths[ed_val:]
                poses = np.concatenate((poses[:st_val], poses[ed_val:]), axis=0)

            # Random Segment
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(sample_times):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path[part].append(x_seg)
                        if not pad_y:
                            Y[part].append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 6))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y[part].append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len[part].append(len(x_seg))
            print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
        
        # Convert to pandas dataframes
        data = {'seq_len': X_len[part], 'image_path': X_path[part], 'pose': Y[part]}
        df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
        # Shuffle through all videos
        if shuffle:
            df = df.sample(frac=1)
        # Sort dataframe by seq_len
        if sort:
            df = df.sort_values(by=['seq_len'], ascending=False)
        df_list.append(df)
    return df_list


class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate len (num of batches, not num of samples)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        self.resize_mode = resize_mode
        self.new_size = (new_sizeize[1], new_sizeize[0])  # cv2 uses (W, H)
        self.img_mean = np.array(img_mean, dtype=np.float32).reshape(3, 1, 1) if img_mean else None
        self.img_std = np.array(img_std, dtype=np.float32).reshape(3, 1, 1) if img_std else None
        self.minus_point_5 = minus_point_5

        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T # opposite rotation of the first frame
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        # groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0:-1]  # get relative pose w.r.t. previois frame

        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0] # get relative pose w.r.t. the first frame in the sequence

        # print('Item before transform: ' + str(index) + '   ' + str(groundtruth_sequence))

        # here we rotate the sequence relative to the first frame
        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]
            # print(location)

        # get relative pose w.r.t. previous frame
        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

        # here we consider cases when rotation angles over Y axis go through PI -PI discontinuity
        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])

        # print('Item after transform: ' + str(index) + '   ' + str(groundtruth_sequence))

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))

        image_sequence = []
        for img_path in image_path_sequence:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.resize_mode == 'crop':
                h, w = img.shape[:2]
                new_h, new_w = self.new_size[1], self.new_size[0]
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                img = img[top:top+new_h, left:left+new_w]
            elif self.resize_mode == 'rescale':
                img = cv2.resize(img, (self.new_size[0], self.new_size[1]))
            img = img.astype(np.float32) / np.float32(255.0)
            if self.minus_point_5:
                img = img - 0.5
            if self.img_mean is not None:
                img = (img - self.img_mean.squeeze()) / self.img_std.squeeze()
            img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW
            img = img.unsqueeze(0)
            image_sequence.append(img)
        image_sequence = torch.cat(image_sequence, 0)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)


class CachedImageSequenceDataset(Dataset):
    """
    Optimized dataset that preloads all images into memory for much faster data loading.
    """
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False, preload=True):
        self.resize_mode = resize_mode
        self.new_size = (new_sizeize[1], new_sizeize[0])  # cv2 uses (W, H)
        self.img_mean = np.array(img_mean, dtype=np.float32).reshape(3, 1, 1) if img_mean else None
        self.img_std = np.array(img_std, dtype=np.float32).reshape(3, 1, 1) if img_std else None
        self.minus_point_5 = minus_point_5

        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

        # Preload images into memory
        self.preload = preload
        if preload:
            print("Preloading images into memory...")
            self._load_all_images()
            print("Images preloaded!")

    def _load_all_images(self):
        """Load all images into memory dictionary keyed by path."""
        from collections import defaultdict
        import cv2

        # Group paths by folder to minimize disk seeks
        folder_paths = defaultdict(list)
        for paths in self.image_arr:
            for p in paths:
                folder_paths[p.split('/')[-2]].append(p)

        self.image_cache = {}
        for folder, paths in folder_paths.items():
            print(f"  Loading folder {folder} ({len(paths)} images)...")
            for i, p in enumerate(paths):
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize immediately to save memory
                if self.resize_mode == 'crop':
                    h, w = img.shape[:2]
                    new_h, new_w = self.new_size[1], self.new_size[0]
                    top = (h - new_h) // 2
                    left = (w - new_w) // 2
                    img = img[top:top+new_h, left:left+new_w]
                elif self.resize_mode == 'rescale':
                    img = cv2.resize(img, (self.new_size[0], self.new_size[1]))
                # Normalize
                img = img.astype(np.float32) / 255.0
                if self.minus_point_5:
                    img = img - 0.5
                if self.img_mean is not None:
                    img = (img - self.img_mean.squeeze()) / self.img_std.squeeze()
                # Convert to CHW tensor format but don't convert to tensor yet
                img = img.transpose(2, 0, 1)  # HWC -> CHW
                self.image_cache[p] = img
                if (i + 1) % 500 == 0:
                    print(f"    Loaded {i+1}/{len(paths)} images...")

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)

        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0]

        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]

        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])

        if self.preload:
            # Use cached images
            image_sequence = []
            for img_path in image_path_sequence:
                img = self.image_cache[img_path].copy()
                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                image_sequence.append(img)
            image_sequence = torch.cat(image_sequence, 0)
        else:
            # Fallback to original loading
            image_sequence = []
            for img_path in image_path_sequence:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.resize_mode == 'crop':
                    h, w = img.shape[:2]
                    new_h, new_w = self.new_size[1], self.new_size[0]
                    top = (h - new_h) // 2
                    left = (w - new_w) // 2
                    img = img[top:top+new_h, left:left+new_w]
                elif self.resize_mode == 'rescale':
                    img = cv2.resize(img, (self.new_size[0], self.new_size[1]))
                img = img.astype(np.float32) / 255.0
                if self.minus_point_5:
                    img = img - 0.5
                if self.img_mean is not None:
                    img = (img - self.img_mean.squeeze()) / self.img_std.squeeze()
                img = torch.from_numpy(img.transpose(2, 0, 1))
                img = img.unsqueeze(0)
                image_sequence.append(img)
            image_sequence = torch.cat(image_sequence, 0)

        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)


# Example of usage
if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    sample_times = 1
    folder_list = ['00']
    seq_len_range = [5, 7]
    df = get_data_info(folder_list, seq_len_range, overlap, sample_times)
    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))
    # Customized Dataset, Sampler
    n_workers = 8
    resize_mode = 'crop'
    new_size = (150, 600)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    dataset = ImageSequenceDataset(df, resize_mode, new_size, img_mean)
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        s, x, y = batch
        print('='*50)
        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))
    
    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)
