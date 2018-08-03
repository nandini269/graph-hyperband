import config
import os
import json
from hyperband import Hyperband
import time
#from model import get_base_model
#from utils import prepare_dirs, save_results

def prepare_dirs(dirs):
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

def save_results(r):
    date = time.strftime("%Y%m%d_%H%M%S")
    filename = date + '_results.json'
    param_path = os.path.join('./results/', filename)
    with open(param_path, 'w') as fp:
        json.dump(r, fp, indent=4, sort_keys=True)


def main(args):
    
    # ensure directories are setup
    dirs = [args.data_dir, args.ckpt_dir]
    prepare_dirs(dirs)

    # instantiate hyperband object
    hyperband = Hyperband(args)

    # tune
    results = hyperband.tune()

    # dump results
    save_results(results)


if __name__ == '__main__':
    args, unparsed = config.get_args()
    main(args)
