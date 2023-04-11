from collections import defaultdict

import os
import time
import datetime

import torch
import torch.nn.functional as F
from transformers import BertModel
from score import score
from torch.utils.tensorboard import SummaryWriter

from models_gan import Generator, Discriminator
from graph_data import get_loaders
import numpy as np
from tqdm import tqdm

class Solver(object):
    """Solver for training and testing LIC-GAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""
        print(config.batch_size)
        # Log
        self.log = log

        # Data loader.
        self.train_data, self.val_data, self.test_data = get_loaders(config.data_dir, 
                                                                     config.N, 
                                                                     config.max_len, 
                                                                     config.lm_model, 
                                                                     config.batch_size,
                                                                     num_workers=1)

        # Model configurations.
        self.N = config.N
        self.z_dim = config.z_dim
        self.mha_dim = config.mha_dim
        self.n_heads = config.n_heads
        self.hid_dims = config.hid_dims
        self.hid_dims_2 = config.hid_dims_2
        self.m_dim = config.m_dim
        self.conv_dim = config.conv_dim
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.post_method = config.post_method
        
        self.lm_model = config.lm_model
        self.max_len = config.max_len

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = len(self.train_data)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        # if self.la > 0:
        #     self.n_critic = config.n_critic
        # else:
        #     self.n_critic = 1
        self.n_critic = config.n_critic
        self.resume_epoch = config.resume_epoch
        self.val_step = config.val_step

        # Training or testing.
        self.mode = config.mode

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)

        # Directories.
        self.log_dir = config.log_dir
        self.writer = SummaryWriter(config.log_dir)
        self.model_dir = config.model_dir

        # Step size.
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.N,
                           self.z_dim,
                           self.hid_dims,
                           self.hid_dims_2,
                           self.mha_dim,
                           self.n_heads,
                           self.dropout)
        self.D = Discriminator(self.N,
                               self.conv_dim, 
                               self.m_dim, 
                               self.mha_dim,
                               self.n_heads,
                               self.dropout)
        self.bert = BertModel.from_pretrained(self.lm_model)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, betas=(0, 0.9))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, betas=(0, 0.9))
        self.g_scheduler = torch.optim.lr_scheduler.LinearLR(self.g_optimizer,
                                                             1.,
                                                             1./self.num_epochs,
                                                             self.num_epochs)
        self.d_scheduler = torch.optim.lr_scheduler.LinearLR(self.d_optimizer,
                                                             1.,
                                                             1./self.num_epochs,
                                                             self.num_epochs)
        self.print_network(self.G, 'G', self.log)
        self.print_network(self.D, 'D', self.log)
        self.print_network(self.bert, self.lm_model, self.log)

        self.G.to(self.device)
        self.D.to(self.device)
        self.bert.to(self.device)
        
    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method, temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
        elif method == 'sigmoid':
            softmax = [F.sigmoid(e_logits / temperature)
                       for e_logits in listify(inputs)] 
        else:
            softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]
        

        return delistify([e for e in (softmax)])

    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch is not None:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Start training.
        if self.mode == 'train':
            print('Start training...')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
                self.g_scheduler.step()
                self.d_scheduler.step()
                # if i == start_epoch:
                #     self.la = 1
        elif self.mode == 'test':
            # assert self.resume_epoch is not None
            self.train_or_valid(epoch_i=start_epoch, train_val_test='test')
        else:
            raise NotImplementedError

    def get_gen_adj_mat(self, adj_mat, method):
        adj_mat = self.postprocess(adj_mat, method)
        adj_mat = torch.nan_to_num(adj_mat, nan=0., posinf=0., neginf=0.)
        adj_mat = (adj_mat + adj_mat.permute(0, 2, 1)) / 2
        adj_mat = torch.round(adj_mat)
        assert torch.all(torch.eq(adj_mat, adj_mat.permute(0, 2, 1)))
        return adj_mat

    # def get_reward(self, n_hat, e_hat, method):
    #     (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
    #     edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]
    #     mols = [self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
    #             for e_, n_ in zip(edges_hard, nodes_hard)]
    #     reward = torch.from_numpy(self.reward(mols)).to(self.device)
    #     return reward

    def save_checkpoints(self, epoch_i):
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(epoch_i + 1))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(epoch_i + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir))

    def train_or_valid(self, epoch_i, train_val_test='val'):
        # The first several epochs using RL to purse stability (not used).
        # if epoch_i < 0:
        #     cur_la = 0
        # else:
        #     cur_la = self.la

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        # Iterations
        the_step = self.num_steps
        if train_val_test == 'val':
            if self.mode == 'train':
                the_step = len(self.val_data)
            print('[Validating]')
        if train_val_test == 'test':
            the_step = len(self.test_data)

        for a_step in tqdm(range(the_step)):
            if train_val_test == 'val':
                adj_mat, ids, mask, desc = next(iter(self.val_data))
                z = self.sample_z(adj_mat.shape[0])
            elif train_val_test == 'test':
                adj_mat, ids, mask, desc = next(iter(self.test_data))
                z = self.sample_z(adj_mat.shape[0])
            elif train_val_test == 'train':
                adj_mat, ids, mask, desc = next(iter(self.train_data))
                z = self.sample_z(self.batch_size)
            else:
                raise NotImplementedError

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            adj_mat = adj_mat.to(self.device)
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            z = torch.from_numpy(z).to(self.device).float()

            # Current steps
            cur_step = self.num_steps * epoch_i + a_step
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            
            # Compute the bert out
            with torch.no_grad():
                bert_out = self.bert(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
            # Compute losses with real inputs.
            if train_val_test != 'train':
                with torch.no_grad():
                    logits_real, features_real = self.D(adj_mat, bert_out)
                    # Z-to-target
                    adjM_logits = self.G(z, bert_out)
            else:
                logits_real, features_real = self.D(adj_mat, bert_out)
                # Z-to-target
                adjM_logits = self.G(z, bert_out)
        
            # Postprocess with Gumbel softmax
            adjM_hat = self.postprocess(adjM_logits, self.post_method)
            if train_val_test != 'train':
                with torch.no_grad():
                    logits_fake, features_fake = self.D(adjM_hat, bert_out)
            else:
                logits_fake, features_fake = self.D(adjM_hat, bert_out)

            # Compute losses for gradient penalty.
            eps = torch.rand(logits_real.size(0), 1, 1).to(self.device)
            x_int0 = (eps * adj_mat + (1. - eps) * adjM_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, bert_out)
            grad_penalty = self.gradient_penalty(grad0, x_int0)

            d_loss_real = torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty

            self.writer.add_scalar(f'{train_val_test}/loss_D', loss_D.item(), cur_step)
            self.writer.add_scalar(f'{train_val_test}/loss_D_real', d_loss_real.item(), cur_step)
            self.writer.add_scalar(f'{train_val_test}/loss_D_fake', d_loss_fake.item(), cur_step)
            self.writer.add_scalar(f'{train_val_test}/loss_D_gp', self.la_gp * grad_penalty.item(), cur_step)

            losses['l_D/R'].append(d_loss_real.item())
            losses['l_D/F'].append(d_loss_fake.item())
            losses['l_D'].append(loss_D.item())

            # Optimise discriminator.
            if train_val_test == 'train' and cur_step % self.n_critic != 0:
                self.reset_grad()
                loss_D.backward()
                self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            # # Z-to-target
            # edges_logits, nodes_logits = self.G(z)
            # # Postprocess with Gumbel softmax
            # (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)
            # logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

            # Value losses
            # value_logit_real, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
            # value_logit_fake, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)

            # Feature mapping losses. Not used anywhere in the PyTorch version.
            # I include it here for the consistency with the TF code.
            # f_loss = (torch.mean(features_real, 0) - torch.mean(features_fake, 0)) ** 2

            # # Real Reward
            # reward_r = torch.from_numpy(self.reward(mols)).to(self.device)
            # # Fake Reward
            # reward_f = self.get_reward(nodes_hat, edges_hat, self.post_method)

            # Losses Update
            loss_G = -logits_fake
            # Original TF loss_V. Here we use absolute values instead of the squared one.
            # loss_V = (value_logit_real - reward_r) ** 2 + (value_logit_fake - reward_f) ** 2
            # loss_V = torch.abs(value_logit_real - reward_r) + torch.abs(value_logit_fake - reward_f)
            # loss_RL = -value_logit_fake

            loss_G = torch.mean(loss_G)
            # loss_V = torch.mean(loss_V)
            # loss_RL = torch.mean(loss_RL)
            losses['l_G'].append(loss_G.item())
            self.writer.add_scalar('loss_G', loss_G.item(), cur_step)
            # losses['l_RL'].append(loss_RL.item())
            # losses['l_V'].append(loss_V.item())

            # alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
            # train_step_G = cur_la * loss_G
            self.writer.add_scalar(f'{train_val_test}/loss_G', loss_G.item(), cur_step)

            # train_step_V = loss_V

            if train_val_test == 'train':
                self.reset_grad()

                # Optimise generator.
                if cur_step % self.n_critic == 0:
                    loss_G.backward(retain_graph=True)
                    self.g_optimizer.step()

                # Optimise value network.
                # if cur_step % self.n_critic == 0:
                #     train_step_V.backward()
                #     self.v_optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Get scores.
            # torch.cuda.empty_cache()
            if self.mode == 'test' or a_step % self.val_step == 0:
                mats = self.get_gen_adj_mat(adjM_hat, self.post_method)
                results = score(desc, mats.detach().cpu().numpy().astype(int))
                for k, v in results.items():
                    scores[k].append(v)
            
                scores_ = []
                for i in range(len(desc)):
                    ok, cnt = 0, 0
                    for k, v in results.items():
                        if v[i] == -1:
                            continue
                        cnt += 1
                        ok += v[i]
                    scores_.append(ok / cnt)
                self.writer.add_scalar(f'{train_val_test}/score', np.mean(scores_), cur_step)

                if a_step + 1 == the_step:
                    # Save checkpoints.
                    if self.mode == 'train':
                        if (epoch_i + 1) % self.model_save_step == 0:
                            self.save_checkpoints(epoch_i=epoch_i)

                    # Print out training information.
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.num_epochs)

                    is_first = True
                    for tag, value in losses.items():
                        if is_first:
                            log += "\n{}: {:.2f}".format(tag, np.mean(value))
                            is_first = False
                        else:
                            log += ", {}: {:.2f}".format(tag, np.mean(value))
                    if self.mode == 'test' or (epoch_i + 1) % self.model_save_step == 0:
                        is_first = True
                        for tag, value in scores.items():
                            if is_first:
                                log += "\n{}: {:.2f}".format(tag, np.mean(value))
                                is_first = False
                            else:
                                log += ", {}: {:.2f}".format(tag, np.mean(value))
                    print(log)

                    if self.log is not None:
                        self.log.info(log)

