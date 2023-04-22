import argparse


def str2bool(v):
    return v.lower() in ['true']


def get_GAN_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lm_model', type=str, default='roberta-base', help='LM model')
    parser.add_argument('--N', type=int, default=50, help='max number of nodes')
    parser.add_argument('--max_len', type=int, default=128, help='max number of tokens input to LM')
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of latent vector')
    parser.add_argument('--mha_dim', type=int, default=768, help='dimension of vectors uses in multi-head attentin')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads to be used in multi-head attention')
    parser.add_argument('--gen_dims', default=[[128, 256, 768], [512, 256, 128]], help='hidden dimensions of MLP layer in G before and after attention')
    parser.add_argument('--disc_dims', default=[[128, 128], [512, 768], [512, 256, 128]], help='hidden dimensions of MLP layer in D before and after attention')
    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty')
    parser.add_argument('--lambda_rew', type=float, default=0, help='weight for reward loss')
    parser.add_argument('--lambda_wgan', type=float, default=1, help='whether or not to use wgan loss')
    parser.add_argument('--post_method', type=str, default='hard_gumbel', choices=['sigmoid', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training D')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='learning rate for D')
    parser.add_argument('--b_lr', type=float, default=1e-5)
    parser.add_argument('--r_lr', type=float, default=1e-3, help='learning rate for R')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=4, help='number of D updates per each G update')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--bert_unfreeze', type=int, default=0)

    # Use either of these two datasets.
    parser.add_argument('--data_dir', type=str, default='data')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--model_save_step', type=int, default=20)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    
    # For training
    config = parser.parse_args()
    config.restore_G = None #'/home/abisheks/10708-Project/MolGAN-PyTorch/results/2023-04-10_14-55/models/40-G.ckpt'
    config.restore_D = None #'/home/abisheks/10708-Project/MolGAN-PyTorch/results/2023-04-10_14-55/models/40-D.ckpt'
    config.restore_R = None
    config.restore_B_D = None
    config.restore_B_G = None

    # Wandb
    config.name = 'symm_fcn_final_ds_gumbel_roberta_adjonly_rew'
    
    # Involve bert unfreeze
    config.bert_unfreeze = 0
    
    # For testing
    # config.mode = 'test'
    # config.saving_dir = 'exp_results/VAE/2020-06-03_13-38-00'
    # config.resume_epoch = 150

    return config


def get_VAE_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128, 256, 512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]],
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training D')
    parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.001, help='learning rate for D')
    parser.add_argument('--r_lr', type=float, default=0.001, help='learning rate for R')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Use either of these two datasets.
    parser.add_argument('--mol_data_dir', type=str, default='data/qm9_5k.sparsedataset')
    # parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='../exp_results/VAE_test/')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    # For training
    config = parser.parse_args()
    config.mode = 'train'
    config.lambda_wgan = 1.0
    config.g_lr = config.d_lr = 1e-4
    config.model_save_step = 1
    config.batch_size = 128
    config.num_epochs = 150

    # For testing
    # config.mode = 'test'
    # config.saving_dir = 'exp_results/VAE/2020-06-03_13-38-00'
    # config.resume_epoch = 150

    return config
