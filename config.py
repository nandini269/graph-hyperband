import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='HyperSearch')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# hyperband params
hyper_arg = add_argument_group('Hyperband Params')
hyper_arg.add_argument('--max_iter', type=int, default=35,                                                      
                       help='Maximum # of iters allocated to a given config')
hyper_arg.add_argument('--eta', type=int, default=3.1,
                       help='Proportion of configs discarded in each round of SH')
# hyper_arg.add_argument('--epoch_scale', type=str2bool, default=True,
#                        help='Compute `max_iter` in terms of epochs or mini-batch iters')          #checkkkkkkk

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--name', type=str, default='mnist',
                      help='Dataset name to train and validate on')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=False,
                      help='Whether to shuffle the train and valid indices')

# optim params
train_arg = add_argument_group('Optim Params')
train_arg.add_argument('--patience', type=int, default=7,
                       help='# of epochs to wait before early stopping')

# misc params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--num_gpu', type=int, default=0,
                      help='0 for cpu, greater for gpu')
misc_arg.add_argument('--data_dir', type=str, default='./data/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--print_freq', type=int, default=10,
                      help='How frequently to print training details')


def get_args():
    args, unparsed = parser.parse_known_args()
    return args, unparsed
