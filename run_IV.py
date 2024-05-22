import jax
jax.config.update('jax_platform_name', 'cpu')

import argparse
from datetime import datetime
from config import Config
from solver import Solver
from time import time

from datasets.data_IV import Toy_data
from bootstrap.bootstrap_IV_closedform import BootstrapMSE

from joblib import Parallel, delayed
import os

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10' 
print("Number of devices: ", jax.devices())

# from jax import config
# config.update("jax_disable_jit", True)  #TODO

###########################################
# Runner code
###########################################

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Parameters for Hyper-param sweep
        parser.add_argument("--base", default=0, help="Base counter for Hyper-param search", type=int)
        parser.add_argument("--inc", default=0, help="Increment counter for Hyper-param search", type=int)
        parser.add_argument("--hyper", default='default', help="Which Hyper param settings")
        parser.add_argument("--seed", default=0, help="seed for variance testing", type=int)

        # General parameters
        parser.add_argument("--save_count", default=10, help="Number of ckpts for saving results and model", type=int)
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",
                            choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=2, type=int, help="Debug modes: {0, 1}")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--save_model", default=False, type=self.str2bool, help="flag to save model ckpts")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        parser.add_argument("--cluster", default=False, help="Running on cluster?", type=self.str2bool)
        parser.add_argument("--invalid_run", default=False, help="RunValid", type=self.str2bool)

        # Book-keeping parameters
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
        parser.add_argument("--experiment", default='boot2SLS', help="Name of the experiment")

        self.Experiment_setup_args(parser)   # Hyper-params that govern what the problem setup is 
        self.Env_n_Agent_args(parser)   # Decide the Environment and the Agent
        self.Main_args(parser)          # Main hyper-parameters that need to be tuned

        self.parser = parser

    def Experiment_setup_args(self, parser):
        parser.add_argument("--n_IV", default=int(5), help="number of IVs", type=int)
        parser.add_argument("--conf_strength", default=0.5, help="1.0 Strength of confounder", type=float)
        parser.add_argument("--total_samples", default=int(1000), help="number of samples per batch", type=int)
        parser.add_argument("--samples_first_batch", default=int(200), help="number of samples per batch", type=int)
        parser.add_argument("--n_batches", default=5, help="number of times data is collected", type=int)
        parser.add_argument("--n_trials", default=1, help="number of times the entire thing is rerun", type=int)

    def Env_n_Agent_args(self, parser):
        parser.add_argument("--algo_name", default='proposed', help="Learning algorithm",  
                            choices=['proposed', 'oracle', 'uniform'])                  
        parser.add_argument("--env_name", default='synthetic', help="Environment to run the code",
                           choices=['synthetic', 'semi-synthetic'] )  
        
    def Main_args(self, parser):        
        parser.add_argument("--sampler_epochs", default=int(1e3), help="Number of training epochs for sampler", type=int)
        parser.add_argument("--sampler_lr_start", default=5e-2, help="5e-2 Starting learning rate", type=float)
        parser.add_argument("--sampler_lr_end", default=1e-3, help="5e-4 Ending learning rate", type=float)
        parser.add_argument("--reset_sampler_param", default=True, help="we reset params between each data collection?", type=self.str2bool)
       
        parser.add_argument("--epochs", default=int(1e3), help="Number of training epochs for sampler", type=int)
        parser.add_argument("--estimator_lr_start", default=1e-1, help="Starting learning rate", type=float)
        parser.add_argument("--estimator_lr_end", default=1e-3, help="Ending learning rate", type=float)
        parser.add_argument("--reset_outer_param", default=True, help="we reset params between each data collection?", type=self.str2bool)
        
        parser.add_argument("--inner_epochs", default=int(5e2), help=" 5e2 Number of inner training epochs for est/nuisance", type=int)
        parser.add_argument("--lr_start", default=1e-2, help="* 1e-1/2 Starting learning rate", type=float)
        parser.add_argument("--lr_end", default=1e-3, help="1e-3 Ending learning rate", type=float)
        parser.add_argument("--reset_inner_param", default=False, help="* True reset inner params between each bootstrap?", type=self.str2bool)
        
        parser.add_argument("--hess_inv", default='CG', help=" [CG or Neumann] ")
        # parser.add_argument("--neumann_alpha", default=1e-0, help="* 1e-0 Scaling factor of Hessian for Neumann series expansion", type=float)
        # parser.add_argument("--neumann_iter", default=5, help="* 5 Number of terms in Neumann series", type=int)
        parser.add_argument("--subsampling_power", default=0.7, help="0.75 Bootstrap subsample size", type=float)
        parser.add_argument("--boot_sample_size", default=30, help="Number of bootstrap dataset replicas", type=int)
        
        parser.add_argument("--inverse_reg", default=1e-5, help="Coeff for compliance regularization", type=float) #UNUSED
        parser.add_argument("--beta_regression", default=True, help="Use regression based (multi)IPW weights", type=self.str2bool)

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser



# @profile
def main(args, inc=-1):
    t = time()

    for trial in range(args.n_trials):
        if inc >=0:
            args.seed = inc 
            
        config = Config(args, trial)

        if config.invalid_run:
            return 0
        
        # Load the data generating process
        if args.env_name == 'semi-synthetic':
            raise ValueError('Not defined Yet')  
        elif args.env_name == 'synthetic':
            config.env = Toy_data  
        else:
            raise ValueError('Unknown env name')     

        config.algo = BootstrapMSE    

        solver = Solver(config=config)

        # with jax.checking_leaks():
        # with jax.profiler.start_trace(config.paths['experiment']):
        solver.train() 
    
    print("Total time taken: {}".format(time()-t))


if __name__ == "__main__":
        # import cProfile
        # cProfile.run('main()', sort='cumtime')
        # main(mode='train')
        args = Parser().get_parser().parse_args()
         
        if args.hyper != 'default':
            # Global flag to set a specific platform, must be used at startup.
            # if args.gpu < 0:

            # FOr running on cluster. Parallelization handled by swarm.
            main(args)
        else:
            # For running the code on laptop. Manual parallelization.
            TRIALS = 10
            _ = Parallel(n_jobs=min(TRIALS, 10))(delayed(main)(args, inc=idx) for idx in range(TRIALS))
