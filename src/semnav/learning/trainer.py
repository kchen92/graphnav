from __future__ import print_function
from __future__ import division

import os
import torch
import torch.optim as optim

from semnav.config import get_config
from semnav.learning import get_net, decode_batch
from semnav.dataset import get_split_datasets
from semnav.dataset.graph_net_frame_dataset import GraphNetFrameDataset
from semnav.learning.behavior_net.behavior_cnn import BehaviorCNN
from semnav.learning.behavior_net.behavior_rnn import BehaviorRNN
from semnav.learning.phase_net.phase_rnn import PhaseRNN
from semnav.learning.graph_net.graph_net import GraphNet
from semnav.dataset.graph_net_dataset import graph_net_collate_fn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


class Trainer(object):

    def __init__(self):
        self.cfg = get_config()
        assert self.cfg.log_dir is not None
        self.evaluator = self.get_evaluator()

    @staticmethod
    def print_train_parameters(cfg):
        print('Device:', cfg.device)
        print('Dataset:', cfg.dataset)
        print('Log directory:', cfg.log_dir)
        print('Batch size:', cfg.batch_size)
        print('Learning rate:', cfg.learning_rate)
        if cfg.ckpt_path is not None:
            print('Checkpoint path:', cfg.ckpt_path)
        print('Number of epochs:', cfg.n_epochs)
        if cfg.behavior_id is not None:
            print('Behavior ID:', cfg.behavior_id)
        print('')

    @staticmethod
    def forward_pass(net, criterion, batch, decode_type, cfg):
        if isinstance(net, BehaviorCNN) or isinstance(net, GraphNet):
            vel, depth = decode_batch(batch, decode_type, cfg)

            # Forward pass
            if isinstance(net, GraphNet):
                graph_net_input, target_output = vel
                outputs = net(depth, graph_net_input)
            else:
                outputs = net(depth)
            loss = criterion(outputs, vel)
        elif isinstance(net, BehaviorRNN) or isinstance(net, PhaseRNN):  # Recurrent network
            seq_len = len(batch)

            # Compute batch size by parsing the batch
            test_vel, _ = decode_batch(batch[0], decode_type, cfg)
            batch_size = test_vel.size(0)

            hidden = net.initial_hidden(batch_size=batch_size)
            # Send to device
            if isinstance(hidden, list):
                hidden = [x.to(cfg.device) for x in hidden]
            else:
                hidden = hidden.to(cfg.device)
            loss = 0.
            for frame_data in batch:
                vel, depth = decode_batch(frame_data, decode_type, cfg)
                output, hidden = net(depth, hidden)
                loss += criterion(output, vel)
            loss /= seq_len  # Compute average loss per timestep
        else:
            raise ValueError('Unrecognized network type.')
        return loss

    def get_decode_type(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    def get_criterion(self):
        raise NotImplementedError('Must be implemented by a subclass.')

    def train(self):
        self.print_train_parameters(self.cfg)

        # Load the network
        net = get_net(self.cfg.behaviornet_type, self.cfg)

        n_val_workers = 1  # Please do not change this
        n_train_workers = self.cfg.n_workers - n_val_workers

        # Load train set and data loader
        train_set, _, _ = get_split_datasets(self.cfg.dataset)
        if self.cfg.dataset_type == 'graph_net':
            train_data_loader = DataLoader(dataset=train_set, batch_size=self.cfg.batch_size,
                                           shuffle=True, num_workers=n_train_workers,
                                           collate_fn=graph_net_collate_fn)
        else:
            train_data_loader = DataLoader(dataset=train_set, batch_size=self.cfg.batch_size,
                                           shuffle=True, num_workers=n_train_workers)

        # Load validation set and data loader
        _, val_set, _ = self.evaluator.load_eval_datasets(self.cfg)
        val_batch_size = self.evaluator.get_evaluation_batch_size()
        if self.cfg.dataset_type == 'graph_net':
            val_data_loader = DataLoader(dataset=val_set, batch_size=val_batch_size, shuffle=False,
                                         num_workers=n_val_workers, collate_fn=GraphNetFrameDataset.collate_fn)
        else:
            val_data_loader = DataLoader(dataset=val_set, batch_size=val_batch_size, shuffle=False,
                                         num_workers=n_val_workers)

        if ((isinstance(net, BehaviorCNN)
                and (self.cfg.dataset_type == 'temporal'))
                or (self.cfg.dataset_type == 'graph_net')):
            decode_type = self.get_decode_type('temporal')
        else:
            decode_type = self.get_decode_type('single_frame')

        criterion = self.get_criterion()
        if self.cfg.gn_cnn_encoder_learning_rate is not None:
            assert self.cfg.affordance_cnn_ckpt_path is not None
            assert self.cfg.freeze_affordance_cnn_weights is not True
            cnn_encoder_parameter_names, cnn_encoder_parameters = net.cnn_encoder.get_params_for_all_but_last_layer()

            all_other_net_parameters = [parameter for name, parameter in net.named_parameters()
                                        if not name.startswith('cnn_encoder')
                                        or (name[12:] not in cnn_encoder_parameter_names)]

            list_of_param_dicts = [
                {'params': cnn_encoder_parameters, 'lr': self.cfg.gn_cnn_encoder_learning_rate},
                {'params': all_other_net_parameters},  # Use default learning rate
                ]

            optimizer = optim.Adam(list_of_param_dicts, lr=self.cfg.learning_rate)
        else:
            optimizer = optim.Adam(net.parameters(), lr=self.cfg.learning_rate)
        iteration = 0
        running_loss = 0.
        with SummaryWriter(self.cfg.log_dir) as writer:
            # Train
            print('Beginning training!')
            for epoch in range(self.cfg.n_epochs):
                for i, batch in enumerate(train_data_loader):
                    net.train()

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    loss = self.forward_pass(net, criterion, batch, decode_type, self.cfg)

                    # Backward pass
                    loss.backward()

                    # Gradient step
                    optimizer.step()

                    # Update counters
                    iteration += 1
                    running_loss += loss.item()

                    # Print progress
                    if iteration % self.cfg.print_freq == 0:
                        train_loss = running_loss / self.cfg.print_freq
                        print('[epoch %4d, iteration in epoch %5d, iteration %6d] loss: %.3f' %
                              (epoch + 1, i + 1, iteration, train_loss))
                        writer.add_scalar('data/train_loss', train_loss, iteration)
                        running_loss = 0.

                    # Run validation
                    if iteration % self.cfg.val_freq == 0:
                        print('running validation...')
                        val_loss = self.evaluator.evaluate_dataset(net, criterion, val_data_loader)
                        print('validation loss: %.3f' % val_loss)
                        writer.add_scalar('data/val_loss', val_loss, iteration)

                    # Save checkpoint
                    if (self.cfg.ckpt_freq is not None) and (iteration % self.cfg.ckpt_freq == 0):
                        ckpt_path = os.path.join(self.cfg.ckpt_dir, 'iteration-%06d.model' % iteration)
                        print('Saving checkpoint:', ckpt_path)
                        torch.save(net.state_dict(), ckpt_path)


if __name__ == '__main__':
    Trainer().train()
