import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorRNN(nn.Module):

    def __init__(self, rnn_type, hidden_size, num_layers):
        super(BehaviorRNN, self).__init__()
        self.is_recurrent = True
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.rnn_type = rnn_type

        if self.rnn_type == 'rnn':
            rnn_class = nn.RNN
        elif self.rnn_type == 'gru':
            rnn_class = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_class = nn.LSTM
        else:
            raise ValueError('Invalid RNN type.')

        # Input resolution: 320x240
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0)
        self.conv6_bn = nn.BatchNorm2d(512)

        rnn_input_size = 512 * 2 * 3
        self.rnn = rnn_class(input_size=rnn_input_size, hidden_size=hidden_size,
                             num_layers=self.n_layers)
        self.fc4 = nn.Linear(hidden_size, 2)

    def forward(self, cur_input, cur_hidden):
        """Forward pass a single input (seq_len == 1) through the CNN-RNN.

        Args:
            cur_input: Input of shape (batch_size x n_channels x height x width). Since we would
                    like to reuse this code for train and test, we only process one input at a time.
                    Thus, seq_len = 1 and the input should be (1 x batch_size x input_size).
            cur_hidden: Current (previous?) hidden state.

        Returns:
            output: Hidden state for each output. It has shape (seq_len x batch_size x hidden_size).
                    Since our seq_len is usually 1, this will generally be of shape
                    (1 x batch_size x hidden_size).
        """
        # CNN encoder
        x = cur_input
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        # x now has size torch.Size([32, 512, 2, 3])

        batch_size = x.size(0)
        x = torch.reshape(x, (batch_size, -1))  # Flatten
        x = torch.unsqueeze(x, dim=0)  # Add a seq_len dimension of size 1

        # RNN
        output, hidden = self.rnn(x, cur_hidden)
        # output should have shape torch.Size([1, batch_size, hidden_size])
        # hidden should have shape torch.Size([self.n_layers, batch_size, hidden_size])
        output = torch.squeeze(output, dim=0)
        output = self.fc4(output)
        return output, hidden

    def initial_hidden(self, batch_size):
        """Initial hidden state. Note that the default hidden state is zeros if not provided.
        """
        if (self.rnn_type == 'rnn') or (self.rnn_type == 'gru'):
            return torch.zeros(self.n_layers, batch_size, self.hidden_size)
        elif self.rnn_type == 'lstm':
            return [torch.zeros(self.n_layers, batch_size, self.hidden_size) for _ in range(2)]  # 2 because cell state and hidden state
