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
import csv
from optparse import OptionParser
from collections import deque
from splitutils import scaffold_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.autograd import Variable
from numpy.random import uniform, normal, randint, choice
from mpn import *


parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
# parser.add_option("-v", "--d_valid", dest="valid_path")
# parser.add_option("-z", "--test", dest="test_path")
parser.add_option("-c", "--metric", dest="metric")
parser.add_option("-a", "--anneal", dest="anneal", default=-1)
# parser.add_option("-r", "--random-seed", type=int, dest="random_seed", default=1013)
# parser.add_option("-b", "--batch", dest="batch_size", default=70)
parser.add_option("-m", "--save_dir", dest="save_path", default='model')
parser.add_option("-s", "--split", dest="split", default='random')
parser.add_option("-r", "--shuffle", dest="shuffle", default=False)                         # change back

opts,args = parser.parse_args()
nan_file = open("record_nan.txt",'w')
final_result = open("final_results.txt",'w')

def create_split(data, seed, notFirst):    # make sure this is fine for float

    if opts.split == 'random':
        if opts.shuffle or notFirst:
            print('random splitting')
            np.random.seed(seed)
            np.random.shuffle(data)
        train_size,test_size = int(len(data) * 0.8), int(len(data) * 0.1)
        train = data[ : train_size]
        valid = data[train_size : train_size + test_size]
        test = data[train_size + test_size : ]
    elif opts.split == 'scaffold':
        print('scaffold splitting')
        train, valid, test = scaffold_split(data)
    return train, valid, test


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
    elif a == 'prelu':
        return nn.PReLU(num_parameters=1, init=0.25)
    elif a == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    else:
        raise ValueError('[!] Unsupported activation.')

def get_default_metric(m):
    if m == 'classify':
        return 'roc'
    else:
        return 'rmse'

def get_data(metric, path):
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
            if metric == 'classify':
                vals = [func(x) for x in vals[1:]]
            else:
                vals = [float(x) for x in vals[1:]]
            data.append((smiles,vals))
    return data

#make sure its correct

params = {
        # '0_dropout': ['uniform', 0.1, 0.5],
        'non_linear': ['choice', ['relu', 'selu', 'elu', 'tanh', 'sigmoid', 'leakyrelu', 'prelu']],
        # '0_l2': ['log_uniform', 1e-1, 2],
        # '2_act': ['choice', ['selu', 'elu', 'tanh', 'sigmoid']],
        # '2_l1': ['log_uniform', 1e-1, 2],
        # '2_hidden': ['quniform', 512, 1000, 1],
        'hidden_size': ['quniform', 120, 500, 20], # *********                                  third number ??
        # 'all_act': ['choice', [[0], ['choice', ['selu', 'elu', 'tanh']]]],
        'dropout': ['choice', [[0], ['uniform', 0.1, 0.5]]],
        # 'all_batchnorm': ['choice', [0, 1]],
        'depth': ['quniform', 1, 12, 1], # *******
        'optim': ['choice', ["adam"]],
        # 'lr': ['uniform', 1e-3, 8e-3],
        'batch_size': ['quniform', 10, 60, 8]  #                                                
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
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        self.args = args
        #self._parse_params(params)

        # initialize hyperband params R eta s_max B
        #self.epoch_scale = args.epoch_scale
        self.max_iter = args.max_iter                                       # R value
        self.eta = args.eta
        self.s_max = int(np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter


        print(
            "[*] max_iter: {}, eta: {}, B: {}".format(self.max_iter, self.eta, self.B)
        )
        final_result.write("[*] max_iter: {}, eta: {}, B: {}".format(self.max_iter, self.eta, self.B))

        # misc params
        self.data_dir = args.data_dir
        self.ckpt_dir = args.ckpt_dir
        self.num_gpu = args.num_gpu
        self.print_freq = args.print_freq

        #device
        # self.device = torch.device("cuda" if self.num_gpu > 0 else "cpu")

        # data params
        # self.data_loader = None ?
        data = get_data(opts.metric, opts.train_path)

        self.train, self.valid, self.test = create_split(data,1, False)
        if opts.split == "random":
            self.train2, self.valid2, self.test2 = create_split(data,41, True)
            self.train3, self.valid3, self.test3 = create_split(data,72, True)

        self.num_tasks = len(self.train[0][1])
        self.anneal_iter = int(opts.anneal)
        self.classify_or_regress = opts.metric
        self.metric = get_default_metric(opts.metric)

        #print "Number of tasks:", num_tasks

        # self.kwargs = {}
        # if self.num_gpu > 0:
        #      self.kwargs = {'num_workers': 1, 'pin_memory': True}
        # if 'batch_size' not in self.optim_params:
        #     self.batch_hyper = False


        # optim params
        self.patience = args.patience



    def tune(self):
        """
        Tune the hyperparameters of the pytorch model
        using Hyperband.
        """
        best_configs = []
        results = {}

        # finite horizon outerloop
        s_num = 0
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
                #continue;
                # Todo: add condition for all models early stopping

                # run each of the n_i configs for r_i iterations
                val_losses = []
                models = []
            
                with tqdm(total=len(T)) as pbar:
                    for t in T:
                        model,val_loss = self.run_config(self.train, self.valid, t, r_i, s_num, False)          # the first model is saved
                        if opts.split == "random":
                            model2, val_loss2 = self.run_config(self.train2, self.valid2, t, r_i, s_num, True)
                            model3, val_loss3 = self.run_config(self.train3, self.valid3, t, r_i, s_num, True)
                            val_loss = (val_loss + val_loss2 + val_loss3)/3.0
                            models.append([model,model2,model3])
                        else:
                            models.append(model)
                        val_losses.append(val_loss)
                        #torch.save(model.state_dict(), opts.save_path + "/model.bests")
                        pbar.update(1)

                # successive halving
                if i < s - 1:
                    if self.classify_or_regress =='classify':
                        sort_loss_idx = np.argsort(val_losses)[::-1][0:int(n_i / self.eta)]
                        T = [T[k] for k in sort_loss_idx] #if not T[k]["early_stopped"]]
                        tqdm.write("Left withh: {}".format(len(T)))
                    else:
                        sort_loss_idx = np.argsort(val_losses)[0:int(n_i / self.eta)]
                        T = [T[k] for k in sort_loss_idx] #  if not T[k]["early_stopped"]]
                        tqdm.write("Left with: {}".format(len(T)))

            # the last iteration of successive halving has the best loss and model
            s_num = s_num +1                                                               # index keeps track of which best model is
                                                                                            # best overall

            if self.classify_or_regress =='classify':
                best_idx = np.argmax(val_losses)
            else:
                 best_idx = np.argmin(val_losses)

            best_configs.append([T[best_idx], models[best_idx], val_losses[best_idx]])   # this is the thing that keeps track across s

        if self.classify_or_regress =='classify':
            best_idx = np.argmax([b[2] for b in best_configs])
        else:
            best_idx = np.argmin([b[2] for b in best_configs])

        best_model = best_configs[best_idx]
        results["val_loss"] = best_model[2]
        #results["params"] = best_model[0]
        results["str"] = best_model[0].__str__()
        if opts.split != "random":
            b_model = best_model[1]
            torch.save(b_model.state_dict(), opts.save_path + "/model.best1")
        else:
            b_model = best_model[1][0]
            b_model2 = best_model[1][1]
            b_model3 = best_model[1][2]
            torch.save(b_model.state_dict(), opts.save_path + "/model.best1")
            torch.save(b_model2.state_dict(), opts.save_path + "/model.best2")
            torch.save(b_model3.state_dict(), opts.save_path + "/model.best3")
        nan_file.close()
        #print(best_idx)
        #print(best_configs[best_idx])
        print(results)
        #b_model.load_state_dict("/model.best" + str(best_idx))  #get the right model.best
        #b_model = torch.load("/model.best" + str(best_idx))
        t_e = self.get_valid_loss(self.test, b_model, best_model[0])
        print("test error1: %.4f" % t_e)
        final_result.write("\ntest error: %.4f\n" % t_e)
        if opts.split == 'random':
            t_e2 = self.get_valid_loss(self.test2, b_model2, best_model[0])
            t_e3 = self.get_valid_loss(self.test3, b_model3, best_model[0])
            mean_te = np.mean([t_e,t_e3,t_e2])
            std_te = np.std([t_e,t_e3,t_e2])
            print("test error2: %.4f" % t_e2)
            print("test error3: %.4f" % t_e3)
            print("mean test error: %.4f" % mean_te)
            print("STD test error: %.4f" % std_te)
            final_result.write("\n mean test error: %.4f\n" % mean_te)
            final_result.write("\n STD test error: %.4f\n" % std_te)
        final_result.write(results["str"])
        return results

    def get_random_config(self):
        """
        Generates random configurations i.i.d
        Returns a dictionary with hyperparameters and their values

        """
        hyperparams = ["hidden_size", "depth", "non_linear", "dropout", "optim", "batch_size"]
        config = {}
        for h in hyperparams:
            space = params[h]
            if h == "non_linear":
                name_act = sample_from(space)
                config[h] = str2act(name_act)
            else:
                config[h] = sample_from(space)

        config["early_stopped"] = False
        config["val_loss"] = -1
        config["ckpt_name"] = str(uuid.uuid4().hex)
        config["max_epoch"] = 1
        return config

    def get_model_from_config(self, config_i):
        """
        Gets model based on config and classify_or_regress

        """
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)

        hidden_size = "hidden_size"
        depth = "depth"
        non_linear = "non_linear"
        dropout = "dropout"
        optim = "optim"
        batch_size = "batch_size"

        encoder = MPN(config_i[hidden_size], config_i[depth], config_i[non_linear], config_i[dropout])

        if self.classify_or_regress == 'classify':
            model = nn.Sequential(
            encoder,
            nn.Linear(config_i[hidden_size], config_i[hidden_size]),
            nn.ReLU(),
            #config_i[non_linear](),
            nn.Linear(config_i[hidden_size], self.num_tasks * 2)
            )
            model.ckpt_name = config_i["ckpt_name"]
            model.early_stopped = False
            return 0, (nn.CrossEntropyLoss(ignore_index=-1).cuda()), (model.cuda())
        else:
            model = nn.Sequential(
            encoder,
            nn.Linear(config_i[hidden_size], config_i[hidden_size]),
            nn.ReLU(),
            nn.Linear(config_i[hidden_size], self.num_tasks)
            )
            model.ckpt_name = config_i["ckpt_name"]
            model.early_stopped = False
            return 1e5, (nn.MSELoss().cuda()), (model.cuda())



    def run_config(self, train, valid, config_i, num_iters, s_num, cross_val):

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


        hidden_size = "hidden_size"
        depth = "depth"
        non_linear = "non_linear"
        dropout = "dropout"
        optim = "optim"


        # get model from config
        min_val_loss, loss_fn, model = self.get_model_from_config(config_i)
        model = model                                                                       # ?

        for param in model.parameters():                                                    # initialize parameters
            if param.dim() == 1:
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal(param)

        if config_i[optim] == 'adam':
            optimizer = Adam(model.parameters(), lr=1e-3)
        else:
            optimizer = SGD(model.parameters(), lr=0.1)                                     # check learning rate!!!!!!!!!!
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        scheduler.step()

        #check for checkpoint
        if cross_val == False:
            try:
                #import pdb; pdb.set_trace()
                ckpt = self._load_checkpoint(model.ckpt_name)                               # picks the first of the models
                model.load_state_dict(ckpt['state_dict'])
                #print("it's used!!!!!!!!!!")
            except FileNotFoundError:
                pass

        counter = 0
        zero = create_var(torch.zeros(1))

        if config_i["early_stopped"]== True:
            return config_i["model"], config_i["val_loss"]                              # CHECK

        for epoch in range(num_iters):
            # train epoch
            self.call_train_epoch(train, scheduler, loss_fn, optimizer, model, config_i)

            scheduler.step()

            # validate epoch
            cur_loss = self.get_valid_loss(valid, model, config_i)

            if cur_loss > 100:
                break

            #print("validation error: %.4f" % cur_loss)
            final_result.write("validation error: %.4f\n" % cur_loss)


            torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))
            if cross_val==False:
                config_i["max_epoch"] = config_i["max_epoch"] + 1
            if self.compare_loss(cur_loss, min_val_loss):
                min_val_loss = cur_loss
                torch.save(model.state_dict(), opts.save_path + "/model.best"+str(s_num))
                config_i["model"] = model
                config_i["val_loss"] = min_val_loss
                counter = 0
                state = {
                'state_dict': model.state_dict(),             # only gets here when it isn't early stopped
                'min_val_loss': min_val_loss,
                }
                self._save_checkpoint(state, model.ckpt_name)
            else:
                counter += 1
            if counter > self.patience:
                tqdm.write("[!] early stopped!!")
                model.early_stopped = True
                config_i["early_stopped"] = True
                return config_i["model"], min_val_loss

        return model, min_val_loss

        # if opts.save_path is not None:
        #     torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))
        #     if cur_loss > best_loss:
        #         best_loss = cur_loss
        #         print("best validation error so far, epoch = %d" % epoch)
        #         torch.save(model.state_dict(), opts.save_path + "/model.best")

        # return best_loss

        # model.load_state_dict(torch.load(opts.save_path + "/model.best"))
        # print "test error: %.4f" % valid_loss(test)

    def compare_loss(self,a,b):
        if self.classify_or_regress == 'classify':
            return (a > b)
        else:
            return (a < b)


    def call_train_epoch(self, train, scheduler, loss_fn, optimizer, model, config_i):
        if self.classify_or_regress == 'classify':
            self.train_epoch_classify(train, scheduler, loss_fn, optimizer, model, config_i)
        else:
            self.train_epoch_regress(train, scheduler, loss_fn, optimizer, model, config_i)


    def train_epoch_classify(self, train, scheduler, loss_fn, optimizer, model, config_i):
        mse,it = 0,0
    #    print("learning rate: %.6f" % scheduler.get_lr()[0])
        for i in range(0, len(train), config_i["batch_size"]):
            batch = train[i:i + config_i["batch_size"]]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)
            labels = create_var(torch.LongTensor(label_batch))
            model.zero_grad()                                                           # need to zero out gradients for each minibatch
            preds = model(mol_batch)
            loss = loss_fn(preds.view(-1,2), labels.view(-1))
            mse += loss.detach()[0]
            it += config_i["batch_size"]
            loss.backward()                                                             # calculate gradient with loss_fn!
            optimizer.step()

            if i % 1000 == 0:
                #pnorm = math.sqrt(sum([p.norm().data[0] ** 2 for p in model.parameters()]))       # anneal
                #gnorm = math.sqrt(sum([p.grad.norm().data[0] ** 2 for p in model.parameters()]))
                #if np.isnan(pnorm):
                #   nan_file.write(config_i.__str__()+ " learning rate: "+ str(scheduler.get_lr()[0])+"\n")
#                print("loss=%.4f,PNorm=%.2f,GNorm=%.2f" % (mse / it, pnorm, gnorm))
                sys.stdout.flush()
                mse,it = 0,0

            if self.anneal_iter > 0 and i % anneal_iter == 0:
                scheduler.step()
                torch.save(model.state_dict(), opts.save_path + "/model.iter-%d-%d" % (epoch,i))

        if self.anneal_iter == -1: #anneal == len(train)
            scheduler.step()

    def train_epoch_regress(self, train, scheduler, loss_fn, optimizer, model, config_i):
        mse,it = 0,0
#        print("learning rate: %.6f" % scheduler.get_lr()[0])
        for i in range(0, len(train), config_i["batch_size"]):
            batch = train[i:i + config_i["batch_size"]]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)
            labels = create_var(torch.Tensor(label_batch))
            model.zero_grad()                                                           # need to zero out gradients for each minibatch
            preds = model(mol_batch).view(-1)
            loss = loss_fn(preds, labels.view(-1))
            mse += loss.data[0] * config_i["batch_size"]
            it += config_i["batch_size"]                                                       # iteration?
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                #pnorm = math.sqrt(sum([p.norm().data[0] ** 2 for p in model.parameters()]))         # anneal
                #gnorm = math.sqrt(sum([p.grad.norm().data[0] ** 2 for p in model.parameters()]))
                #if np.isnan(pnorm):
                #   nan_file.write(config_i.__str__()+ " learning rate: "+ str(scheduler.get_lr()[0])+"\n")
#                print("loss=%.4f,PNorm=%.2f,GNorm=%.2f" % (mse / it, pnorm, gnorm))
                sys.stdout.flush()
                mse,it = 0,0


    def get_valid_loss(self, data, model, config_i):
        if self.classify_or_regress == 'classify':
            return self.valid_loss_classify(data, model, config_i)
        else:
            return self.valid_loss_regress(data, model, config_i)

    def valid_loss_classify(self, data, model, config_i):
        model.train(False)
        all_preds = [[] for i in range(self.num_tasks)]
        all_labels = [[] for i in range(self.num_tasks)]

        for k in range(0, len(data), config_i["batch_size"]):
            batch = data[k:k+ config_i["batch_size"]]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)

            preds = F.softmax(model(mol_batch).view(-1,self.num_tasks,2), dim=2)                 # 2..is softmax fixed???
            for i in range(self.num_tasks):
                for j in range(len(batch)):
                    if label_batch[j][i] >= 0:
                        all_preds[i].append(preds.data[j][i][1])
                        all_labels[i].append(label_batch[j][i])

        model.train(True)

        #compute roc-auc
        if self.metric == 'roc':                                                            # metric depends on class or regress
            res = []
            for i in range(self.num_tasks):
                if sum(all_labels[i]) == 0: continue
                if min(all_labels[i]) == 1: continue
                try:
                    score = roc_auc_score(all_labels[i], all_preds[i])
                except ValueError:
                    score = 0
                res.append(score)
            return sum(res) / len(res)

        #compute prc-auc
        res = 0
        for i in range(self.num_tasks):
            r,p,t = precision_recall_curve(all_labels[i], all_preds[i])
            val = auc(p,r)
            if math.isnan(val): val = 0
            res += val
        return res / self.num_tasks

    def valid_loss_regress(self, data, model, config_i):
        if self.metric == 'mae':                                                                # matric is hard coded!
            val_loss = nn.L1Loss(reduce=False)
        else:
            val_loss = nn.MSELoss(reduce=False)
        err = torch.zeros(self.num_tasks)
        model.train(False)
        for i in range(0, len(data), config_i["batch_size"]):
            batch = data[i:i+config_i["batch_size"]]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)
            labels = create_var(torch.Tensor(label_batch))

            preds = model(mol_batch)
            loss = val_loss(preds, labels)
            err += loss.data.sum(dim=0).cpu()

        model.train(True)
        err = err / len(data)
        if math.isnan(err): return 99999999999.0
        if self.metric == 'rmse':
            return float(err.sqrt().sum() / self.num_tasks)
        else:
            return float(err.sum() / self.num_tasks)


    def _save_checkpoint(self, state, name):
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)


    def _load_checkpoint(self, name):
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        return ckpt
