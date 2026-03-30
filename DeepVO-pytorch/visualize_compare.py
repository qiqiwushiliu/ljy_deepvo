import matplotlib.pyplot as plt
import numpy as np
import os
from config.params import par

pose_GT_dir = par.pose_dir

# Create result directory if not exists
if not os.path.exists('./test'):
    os.makedirs('./test')
if not os.path.exists('/home/LXT/LJY/DeepVO-pytorch/results/compare'):
    os.makedirs('/home/LXT/LJY/DeepVO-pytorch/results/compare')

# Video list to compare
# video_list = ['04', '05', '07', '10', '09']
video_list = ['00', '02', '08', '09','01', '04', '05', '06', '07', '10']


def plot_overlay(gt, out_lstm, out_cfc):
    """Plot all trajectories overlaid on one figure for direct comparison."""
    x_idx, y_idx = 3, 5

    plt.figure(figsize=(10, 10))
    plt.plot(gt[:, x_idx], gt[:, y_idx], color='g', linewidth=2, label='Ground Truth')
    plt.plot(out_lstm[:, x_idx], out_lstm[:, y_idx], color='r', linewidth=2, label='DeepVO (LSTM)')
    plt.plot(out_cfc[:, x_idx], out_cfc[:, y_idx], color='b', linewidth=2, label='DeepVO (CfC/NCP)')
    plt.scatter([gt[0][x_idx]], [gt[0][y_idx]], marker='s', color='k', s=100, label='Start', zorder=5)
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def load_result(predicted_result_dir, video):
    """Load predicted result from txt file."""
    pose_result_path = '{}out_{}.txt'.format(predicted_result_dir, video)
    with open(pose_result_path) as f_out:
        out = [l.split('\n')[0] for l in f_out.readlines()]
        for i, line in enumerate(out):
            out[i] = [float(v) for v in line.split(',')]
        return np.array(out)


for video in video_list:
    print('='*50)
    print('Video {}'.format(video))

    # Load ground truth
    GT_pose_path = '{}{}.npy'.format(pose_GT_dir, video)
    gt = np.load(GT_pose_path)
    print('GT shape:', gt.shape)

    # Load LSTM result
    out_lstm = load_result('/home/LXT/LJY/DeepVO-pytorch/results/test_lstm/', video)
    print('LSTM result shape:', out_lstm.shape)

    # Load CfC result
    out_cfc = load_result('/home/LXT/LJY/DeepVO-pytorch/results/test_cfc/', video)
    print('CfC result shape:', out_cfc.shape)

    # Calculate errors
    mse_rotate_lstm = 100 * np.mean((out_lstm[:, :3] - gt[:, :3])**2)
    mse_translate_lstm = np.mean((out_lstm[:, 3:] - gt[:, 3:6])**2)
    mse_rotate_cfc = 100 * np.mean((out_cfc[:, :3] - gt[:, :3])**2)
    mse_translate_cfc = np.mean((out_cfc[:, 3:] - gt[:, 3:6])**2)

    print('LSTM - mse_rotate: {:.6f}, mse_translate: {:.6f}'.format(mse_rotate_lstm, mse_translate_lstm))
    print('CfC   - mse_rotate: {:.6f}, mse_translate: {:.6f}'.format(mse_rotate_cfc, mse_translate_cfc))

    # Plot overlay
    plt = plot_overlay(gt, out_lstm, out_cfc)
    plt.savefig('/home/LXT/LJY/DeepVO-pytorch/results/compare/compare_{}.png'.format(video))
    print('Saved: /home/LXT/LJY/DeepVO-pytorch/results/compare/compare_{}.png'.format(video))
    plt.close()

print('='*50)
print('All comparison plots saved!')
