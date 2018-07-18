from torch.optim import SGD, Adam
from torch.autograd import Variable
from numpy.random import uniform, normal, randint, choice

params = {
        # '0_dropout': ['uniform', 0.1, 0.5],
        'non_linear': ['choice', ['relu', 'selu', 'elu', 'tanh', 'sigmoid', 'leakyrelu', 'tanhshrink','prelu']],
        # '0_l2': ['log_uniform', 1e-1, 2],
        # '2_act': ['choice', ['selu', 'elu', 'tanh', 'sigmoid']],
        # '2_l1': ['log_uniform', 1e-1, 2],
        # '2_hidden': ['quniform', 512, 1000, 1],                                                               # add hidden
        'hidden_size': ['quniform', 128, 600, 12], # *********
        # 'all_act': ['choice', [[0], ['choice', ['selu', 'elu', 'tanh']]]],
        'dropout': ['choice', [[0], ['uniform', 0.1, 0.5]]],
        # 'all_batchnorm': ['choice', [0, 1]],
        'depth': ['quniform', 1, 12, 1], # *******                                              
        'optim': ['choice', ["adam", "sgd"]],                                                                  
        # 'lr': ['uniform', 1e-3, 8e-3],
        'batch_size': ['quniform', 10, 100, 5]  # *********
        }

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
        return config

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