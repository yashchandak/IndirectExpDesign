import sys
from yaml import dump
from os import path
import numpy as np
from collections import OrderedDict
from utils import utils
import importlib


class Config(object):
    def __init__(self, args, trial=0):

        # SET UP PATHS
        self.paths = OrderedDict()
        # self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..'))  # Go to project's root
        self.paths['root'] = path.abspath(path.dirname(__file__))  # Go to project's root

        # Do Hyper-parameter sweep, if needed
        self.idx = args.base + args.inc
        if self.idx >= 0 and args.hyper != 'default':
            # hyper_path =  'cluster.Hyper.random_search_{}'.format(args.hyper)
            hyper_path =  'cluster.Hyper.grid_search_{}'.format(args.hyper)
            print('Hyper-params loaded from: ', hyper_path)
            self.hyperparam_sweep = importlib.import_module(hyper_path)
            self.hyperparam_sweep.set(args, self.idx)
            del self.hyperparam_sweep  # *IMP: CanNOT deepcopy, if needed, an object having reference to an imported library\


        # Make results reproducible
        seed = args.seed + trial
        self.seed = seed
        np.random.seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        if self.invalid_run:
            return

        # Frequency of saving results and models.
        # self.save_after = args.max_episodes // args.save_count if args.max_episodes > args.save_count else args.max_episodes

        # add path to models
        if args.cluster:
            # On cluster, do not create separate folders for each seed
            self.paths['experiment'] = path.join(self.paths['root'], 'experiments', args.experiment, args.algo_name + '_' + args.env_name, args.folder_suffix)
            self.tag = '{}_'.format(args.seed)
        else:
            self.paths['experiment'] = path.join(self.paths['root'], 'experiments', args.experiment, args.algo_name  + '_' + args.env_name, args.folder_suffix,  str(args.seed))
            self.tag = ''

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                utils.create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)


        # Output logging
        sys.stdout = utils.Logger(self.paths['experiment'], args.restore, args.log_output)

        # GPU
        # self.device = torch.device('cuda' if args.gpu else 'cpu')

        if self.samples_first_batch <= 0:
            self.samples_per_batch = int(self.total_samples / self.n_batches)
            self.samples_first_batch = self.samples_per_batch
        else:
            self.samples_per_batch = int( (self.total_samples - self.samples_first_batch) / (self.n_batches - 1))

        # Load the data generating process and the algorithm
        # Need to be overwritten
        self.env = None      
        self.algo = None    

        self.tag = self.tag + str(args.n_IV) #+ "_uniform"


        print("=====Configurations=====\n", args)

