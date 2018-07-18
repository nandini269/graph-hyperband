import os
import uuid
import time
import numpy as np

from tqdm import tqdm
#from data_loader import get_train_valid_loader
#from utils import find_key, sample_from, str2act

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import math, random, sys
from optparse import OptionParser
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.autograd import Variable
from numpy.random import uniform, normal, randint, choice
from mpn import *

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--d_valid", dest="valid_path")
parser.add_option("-z", "--test", dest="test_path")
parser.add_option("-c", "--metric", dest="metric", default='roc')
parser.add_option("-a", "--anneal", dest="anneal", default=-1)
parser.add_option("-s", "--random-seed", type=int, dest="random_seed", default=1013)
parser.add_option("-b", "--batch", dest="batch_size", default=50)
opts,args = parser.parse_args()

def find_key(params, partial_key):
    """
    Returns the parameter space or the value from params (dictionary)
    where the partial key is in key .
    """
    return next(v for k, v in params.items() if partial_key in k)


def sample_from(space):                                                                         
    """
    Sample a hyperparameter value from a distribution
    defined and parametrized in the list `space`.
    """
    if type(space)!= list:
        return space
    else:
        distrs = {
            'choice': choice,
            'randint': randint,
            'uniform': uniform,
            'normal': normal,
        }
        s = space[0]
        if type(s)!= str:
            return s
        np.random.seed(int(time.time() + np.random.randint(0, 300)))
        log = s.startswith('log_')
        s = s[len('log_'):] if log else s
        quantized = s.startswith('q')
        s = s[1:] if quantized else s
        distr = distrs[s]
        if s == 'choice':
            return (sample_from (distr(space[1])))
        samp = distr(space[1], space[2])
        if type(samp)== list:
            return sample_from(samp)
        else:
            if log:
                samp = np.exp(samp)
            if quantized:
                samp = round((samp / space[3]) * space[3])
            return samp


def str2act(a):
    """
    Converts the string into the activation function.
    """
    if a == 'relu':
        return nn.ReLU()
    elif a == 'selu':
        return nn.SELU()
    elif a == 'elu':
        return nn.ELU()
    elif a == 'tanh':
        return nn.Tanh()
    elif a == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('[!] Unsupported activation.')

def get_data(path):
    """
    Gets data from file path.
    """

    data = []
    func = lambda x : int(float(x)) if x != '' else -1
    with open(path) as f:
        f.readline()
        for line in f:
            vals = line.strip("\r\n ").split(',')
            smiles = vals[0]
            vals = [func(x) for x in vals[1:]]
            data.append((smiles,vals))
    np.random.shuffle(data)                                                                                     # shuffle data
    return data



params = {
        # '0_dropout': ['uniform', 0.1, 0.5],
        'non_linear': ['choice', ['relu', 'selu', 'elu', 'tanh', 'sigmoid']],
        # '0_l2': ['log_uniform', 1e-1, 2],
        # '2_act': ['choice', ['selu', 'elu', 'tanh', 'sigmoid']],
        # '2_l1': ['log_uniform', 1e-1, 2],
        # '2_hidden': ['quniform', 512, 1000, 1],                                                               # add hidden
        'hidden_size': ['quniform', 128, 512, 1],
        # 'all_act': ['choice', [[0], ['choice', ['selu', 'elu', 'tanh']]]],
        'dropout': ['choice', [[0], ['uniform', 0.1, 0.5]]],
        # 'all_batchnorm': ['choice', [0, 1]],
        'depth': ['quniform', 18, 45, 1],                                               
        'optim': ['choice', ["adam", "sgd"]],                                                                  
        # 'lr': ['uniform', 1e-3, 8e-3],
        # 'batch_size': ['quniform', 32, 128, 1]
        }

class Hyperband(object):
    """
    Hyperband, a bandit-based configuration
    evaluation for hyperparameter optimization [1].

    Hyperband is a principled early-stoppping method
    that adaptively allocates resources to randomly
    sampled configurations, quickly eliminating poor
    ones, until a single configuration remains.

    References
    ----------
    - [1]: Li et. al., https://arxiv.org/abs/1603.06560
    """
    def __init__(self, args):
        """
        Initialize the Hyperband object.

        Args
        ----
        - args: object containing command line arguments.
        - model: the `Sequential()` model you wish to tune.
        - params: a dictionary where the key is the hyperparameter
          to tune, and the value is the space from which to randomly
          sample it.
        """
        self.args = args
        #self._parse_params(params)

        # initialize hyperband params R eta s_max B
        self.epoch_scale = args.epoch_scale
        self.max_iter = args.max_iter                                       # R value
        self.eta = args.eta
        self.s_max = int(np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter

        print(
            "[*] max_iter: {}, eta: {}, B: {}".format(
                self.max_iter, self.eta, self.B
            )
        )

        # misc params
        self.data_dir = args.data_dir
        self.ckpt_dir = args.ckpt_dir
        self.num_gpu = args.num_gpu
        self.print_freq = args.print_freq

        # device
        #self.device = torch.device("cuda" if self.num_gpu > 0 else "cpu")

        # data params
        # self.data_loader = None ?
        self.train = get_data(opts.train_path)                                                          # data shuffled each time
        self.valid = get_data(opts.valid_path)
        self.test = get_data(opts.test_path)
        self.num_tasks = len(self.train[0][1])
        self.anneal_iter = int(opts.anneal)
        self.batch_size = int(opts.batch_size)
        #print "Number of tasks:", num_tasks

        # self.kwargs = {}                                                                        
        # if self.num_gpu > 0:
        #     self.kwargs = {'num_workers': 1, 'pin_memory': True}
        # if 'batch_size' not in self.optim_params:
        #     self.batch_hyper = False


        # optim params                                                                      
        # self.def_optim = args.def_optim
        # self.def_lr = args.def_lr
        self.patience = args.patience

   

    def tune(self):
        """
        Tune the hyperparameters of the pytorch model
        using Hyperband.
        """
        best_configs = []
        results = {}

        # finite horizon outerloop
        for s in reversed(range(self.s_max + 1)):
            # initial number of configs
            n = int(
                np.ceil(
                    int(self.B / self.max_iter / (s + 1)) * self.eta ** s
                )
            )
            # initial number of iterations to run the n configs for
            r = self.max_iter * self.eta ** (-s)

            # finite horizon SH with (n, r)
            T = [self.get_random_config() for i in range(n)]

            tqdm.write("s: {}".format(s))

            for i in range(s + 1):
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))

                tqdm.write(
                    "[*] {}/{} - running {} configs for {} iters each".format(
                        i+1, s+1, len(T), r_i)
                )

                # Todo: add condition for all models early stopping

                # run each of the n_i configs for r_i iterations
                val_losses = []
                with tqdm(total=len(T)) as pbar:                                                            
                    for t in T:                                     # t is a configuration
                        val_loss = self.run_config(t, r_i)          # valid_loss
                        val_losses.append(val_loss)
                        pbar.update(1)                                                  

                # remove early stopped configs and keep the best n_i / eta                                  .. (why ?)
                if i < s - 1:
                    sort_loss_idx = np.argsort(
                        val_losses
                    )[0:int(n_i / self.eta)]
                    T = [T[k] for k in sort_loss_idx if not T[k].early_stopped]
                    tqdm.write("Left with: {}".format(len(T)))

            best_idx = np.argmin(val_losses)
            best_configs.append([T[best_idx], val_losses[best_idx]])

        best_idx = np.argmin([b[1] for b in best_configs])
        best_model = best_configs[best_idx]
        results["val_loss"] = best_model[1]
        results["params"] = best_model[0].new_params
        results["str"] = best_model[0].__str__()
        return results

    def get_random_config(self):
        """
        Generates random configurations i.i.d
        Returns a dictionary with hyperparameters and their values

        """

        hyperparams = ["hidden_size", "depth", "non_linear", "dropout", "optim"]
        config = {}
        for h in hyperparams:
            space = params[h]
            if h == "non_linear":
                name_act = sample_from(space)
                config[h] = str2act(name_act)
            else:
                config[h] = sample_from(space)
        return config


    def run_config(self, config_i, num_iters): 

        """
        Train a particular hyperparameter configuration for a
        given number of iterations and evaluate the loss on the
        validation set.

        For hyperparameters that have previously been evaluated,
        resume from a previous checkpoint.

        Args
        ----
        - config_i: (maybe) dictionary with fields for hyperparameters
        - num_iters: an int indicating the number of iterations
          to train the model for.

        Returns
        -------
        - val_loss: the lowest validaton loss achieved.
        """

        # check point???? early stopped???
        # do we redefine the model every time we call run_config?

        hidden_size = "hidden_size"
        depth = "depth"
        non_linear = "non_linear"
        dropout = "dropout"
        optim = "optim"

        
        # get model from config 
        encoder = MPN(config_i[hidden_size], config_i[depth], config_i[non_linear], dropout=config_i[dropout])

        # print(config_i[non_linear])
        # print('weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        model = nn.Sequential(                                                              
        encoder,
        nn.Linear(config_i[hidden_size], config_i[hidden_size]),
        nn.ReLU(),                                                                                                   
        #config_i[non_linear](), 
        nn.Linear(config_i[hidden_size], self.num_tasks * 2)
        )
        model.ckpt_name = str(uuid.uuid4().hex)
        model.early_stopped = False

        min_val_loss = 0                                                                    # min_val_loss small for classify, big for regress


        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)                                      # different loss function for train and regress
        if config_i[optim] == 'adam':   
            optimizer = Adam(model.parameters(), lr=1e-3)
        else:
            optimizer = SGD(model.parameters(), lr=0.1)                                     # check learning rate!!!!!!!!!!!!!!!
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()

        # check for checkpoint
        try:
            ckpt = self._load_checkpoint(model.ckpt_name)
            model.load_state_dict(ckpt['state_dict'])
        except FileNotFoundError:
            pass

        counter = 0
        zero = create_var(torch.zeros(1))
        for epoch in range(num_iters):                                                  

            # train epoch
            self.train_epoch(scheduler, loss_fn, optimizer, model)                        

            # validate epoch
            cur_loss = self.valid_loss(self.valid, model)
            print("validation error: %.4f" % cur_loss)

            if cur_loss < min_val_loss:                                                     # < classification, > regression
                min_val_loss = cur_loss
                counter = 0
            else:
                counter += 1
            if counter > self.patience:
                tqdm.write("[!] early stopped!!")
                model.early_stopped = True
                return min_val_loss


        state = {
            'state_dict': model.state_dict(),
            'min_val_loss': min_val_loss,
        }
        self._save_checkpoint(state, model.ckpt_name)

        return min_val_loss

        # if opts.save_path is not None:
        #     torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))
        #     if cur_loss > best_loss:
        #         best_loss = cur_loss
        #         print("best validation error so far, epoch = %d" % epoch)
        #         torch.save(model.state_dict(), opts.save_path + "/model.best")

        # return best_loss

        # model.load_state_dict(torch.load(opts.save_path + "/model.best"))
        # print "test error: %.4f" % valid_loss(test)

    def train_epoch(self, scheduler, loss_fn, optimizer, model):
        mse,it = 0,0
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        for i in range(0, len(self.train), self.batch_size):
            batch = self.train[i:i + self.batch_size]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)
            labels = create_var(torch.LongTensor(label_batch))                  
            model.zero_grad()                                                           # need to zero out gradients for each minibatch
            preds = model(mol_batch)                                                    
            loss = loss_fn(preds.view(-1,2), labels.view(-1))                           
            mse += loss                                                                 # mean squared error
            it += self.batch_size                                                       # iteration?
            loss.backward()                                                             # calculate gradient with loss_fn!
            optimizer.step()                                                            

            if i % 1000 == 0:
                pnorm = math.sqrt(sum([p.norm().data[0] ** 2 for p in model.parameters()]))         # anneal
                gnorm = math.sqrt(sum([p.grad.norm().data[0] ** 2 for p in model.parameters()]))
                print("loss=%.4f,PNorm=%.2f,GNorm=%.2f" % (mse / it, pnorm, gnorm))
                sys.stdout.flush()
                mse,it = 0,0

            if self.anneal_iter > 0 and i % anneal_iter == 0:
                scheduler.step()
                torch.save(model.state_dict(), opts.save_path + "/model.iter-%d-%d" % (epoch,i))    

        if self.anneal_iter == -1: #anneal == len(train)
            scheduler.step()

    def valid_loss(self, data, model):                                                     
        model.train(False)
        all_preds = [[] for i in range(self.num_tasks)]
        all_labels = [[] for i in range(self.num_tasks)]

        for k in range(0, len(data), self.batch_size):
            batch = data[k:k+ self.batch_size]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)

            preds = F.softmax(model(mol_batch).view(-1,self.num_tasks,2), dim=2)                 # why 2? is softmax fixed???
            for i in range(self.num_tasks):
                for j in range(len(batch)):
                    if label_batch[j][i] >= 0:
                        all_preds[i].append(preds.data[j][i][1])
                        all_labels[i].append(label_batch[j][i])

        model.train(True)                                                                 

        #compute roc-auc
        if opts.metric == 'roc':                                                            # metric depends on class or regress
            res = []
            for i in range(self.num_tasks):
                if sum(all_labels[i]) == 0: continue
                if min(all_labels[i]) == 1: continue
                res.append( roc_auc_score(all_labels[i], all_preds[i]) )
            return sum(res) / len(res)                                                     

        #compute prc-auc
        res = 0
        for i in range(self.num_tasks):
            r,p,t = precision_recall_curve(all_labels[i], all_preds[i])
            val = auc(p,r)
            if math.isnan(val): val = 0
            res += val
        return res / self.num_tasks


    def _save_checkpoint(self, state, name):
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)


    def _load_checkpoint(self, name):
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        return ckpt






    