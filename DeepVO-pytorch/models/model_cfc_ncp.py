import torch
import torch.nn as nn
from config.params import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from models.ncps.wirings.wirings import NCP
from models.ncps.torch.cfc import CfC


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )


class DeepVOCfC(nn.Module):
    """DeepVO model with CfC (Cellular First-order Continous) neural network and NCP wiring.

    Replaces the LSTM in the original DeepVO with a Neural Circuit Policy (NCP) based CfC.
    The CfC is a continuous-time neural network that is more biologically plausible and
    can be more efficient for sequential visual odometry tasks.
    """
    def __init__(self, imsize1, imsize2, batchNorm=True, ncp_config=None):
        super(DeepVOCfC, self).__init__()
        # CNN - Using the same architecture as CFC_NCP.py for consistency
        self.batchNorm = batchNorm
        self.clip = par.clip

        # CNN layers (same as original DeepVO)
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Compute CNN output shape
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        cnn_output_size = int(np.prod(__tmp.size()))
        print(f"CfC CNN output size: {cnn_output_size}")

        # Dense layer (adapted from CFC_NCP.py)
        self.dense = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(cnn_output_size, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        # CfC with NCP wiring (Neural Circuit Policy)
        # Default NCP configuration if not provided
        if ncp_config is None:
            ncp_config = {
                'input_size': 50,
                'output_size': 10,
                'inter_neurons': 50,
                'command_neurons': 20,
                'motor_neurons': 10,
                'sensory_fanout': 24,
                'inter_fanout': 10,
                'recurrent_command': 10,
                'motor_fanin': 6
            }

        self.wiring = NCP(
            ncp_config['inter_neurons'],      # 50
            ncp_config['command_neurons'],     # 20
            ncp_config['motor_neurons'],       # 10
            ncp_config['sensory_fanout'],      # 24
            ncp_config['inter_fanout'],        # 10
            ncp_config['recurrent_command'],   # 10
            ncp_config['motor_fanin']          # 6
        )

        self.cfc = CfC(
            ncp_config['input_size'],
            self.wiring,
            batch_first=True,
            mixed_memory=True
        )

        # Output layer (6 DoF pose: 3 rotation + 3 translation)
        self.linear = nn.Linear(ncp_config['output_size'], 6)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def encode_image(self, x):
        """CNN encoder for images (same as original DeepVO)."""
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def forward(self, x, hidden_state=None):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, channel, width, height)
            hidden_state: Optional hidden state for LSTM-style hidden state passing

        Returns:
            out: Predicted pose of shape (batch, seq_len, 6)
            hidden_state: Hidden state (for compatibility, returns None for CfC)
        """
        # Stack consecutive images for motion estimation (optical flow style input)
        # x: (batch, seq_len, channel, width, height)
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)  # Consecutive frame pairs

        batch_size = x.size(0)
        seq_len = x.size(1)

        # CNN
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size * seq_len, -1)

        # Dense layers
        x = self.dense(x)
        x = x.view(batch_size, seq_len, -1)

        # CfC (Neural Circuit Policy)
        out, hidden_state = self.cfc(x, hidden_state)

        # Output projection to 6 DoF pose
        out = self.linear(out)

        return out, hidden_state

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y):
        """Calculate weighted MSE loss.

        Args:
            x: Input sequence (batch, seq_len, channel, width, height)
            y: Ground truth pose (batch, seq_len, 6) - relative pose w.r.t. previous frame

        Returns:
            loss: Weighted MSE loss (rotation weighted more than translation)
        """
        predicted, _ = self.forward(x)
        # y shape: (batch, seq, dim_pose) where dim_pose=6 (3 rotation + 3 translation)
        # For training we predict pose for seq_len-1 (skip first frame as reference)
        y = y[:, 1:, :]  # (batch, seq-1, 6)

        # Weighted MSE Loss - rotation angles weighted more than translation
        angle_loss = torch.nn.functional.mse_loss(predicted[:, :, :3], y[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:, :, 3:], y[:, :, 3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        """Training step with gradient clipping.

        Args:
            x: Input sequence
            y: Ground truth pose
            optimizer: PyTorch optimizer

        Returns:
            loss: Loss value for monitoring
        """
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm(self.cfc.parameters(), self.clip)

        optimizer.step()
        return loss


# Default configuration for NCP/CfC (matching CFC_NCP.py: NCP(50,20,10,24,10,10,6))
default_ncp_config = {
    'input_size': 50,
    'output_size': 10,
    'inter_neurons': 50,
    'command_neurons': 20,
    'motor_neurons': 10,
    'sensory_fanout': 24,
    'inter_fanout': 10,
    'recurrent_command': 10,
    'motor_fanin': 6
}

