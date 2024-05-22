# from __future__ import print_function
# from memory_profiler import profile

import numpy as np
from utils import utils
from time import time
from os import path
import jax.numpy as jnp
import matplotlib.pyplot as plt


from tensorboardX import SummaryWriter

class Solver:

    def __init__(self, config):
        # Initialize the required variables
        self.config = config
        self.writer = SummaryWriter(self.config.paths['experiment']) if self.config.debug else None

        self.env = config.env(config=config)
        self.algo = config.algo(config=config, writer=self.writer)
        
    def train(self):
        
        mses = np.zeros((self.config.n_trials, self.config.n_batches, 2))
        losses = np.zeros((self.config.n_trials, self.config.n_batches))
        all_preds = []
        
        title = self.config.tag
        
        for trial in range(self.config.n_trials):
            np.random.seed(self.config.seed + trial)
            self.env.reset()

            train_data, test_data = self.env.get_data(n=self.config.samples_first_batch, 
                                                      uniform=True,
                                                      test=True) 
            mse_train_data = train_data
            self.algo.reset(train_data)

            for batch_idx in range(self.config.n_batches):
                if self.config.reset_outer_param: self.algo.reset(train_data)
                # Compute MSE before updating sampler
                # It allows caching some variables and re-suing them during the update later
                mses[trial, batch_idx, 0] = train_data[0].shape[0]
                mses[trial, batch_idx, 1], pred  = self.algo.get_mse(mse_train_data, test_data)

                if batch_idx < self.config.n_batches - 1:
                    losses[trial, batch_idx] = self.algo.update_sampler(train_data, test_data[0])
                    
                    # Sample data using the provided sampler
                    new_data = self.env.get_data(n=self.config.samples_per_batch, 
                                                sampler=self.algo.get_sampler(),
                                                uniform=self.config.algo_name=='uniform')    

                    # Merge the new data with the old data
                    train_data = [np.vstack([train_data[idx], new_data[idx]]) for idx in range(len(new_data))]

                    mse_train_data = [np.vstack([mse_train_data[idx], new_data[idx]]) for idx in range(len(new_data))]

                    if self.config.debug: self.writer.add_scalar('solver/Estimated MSE', losses[trial, batch_idx], (batch_idx+1)*self.config.samples_per_batch)
                
                if self.config.debug: self.writer.add_scalar('solver/True MSE', mses[trial, batch_idx, 1], (batch_idx+1)*self.config.samples_per_batch)                

            # Check the preictions at maximum batch size
            all_preds.append(pred.reshape(-1))  
            print("===================== Trial {}/{} Completed =====================".format(trial+1, self.config.n_trials))

        if self.config.debug: self.writer.close()

        np.save(path.join(self.config.paths['experiment'], '{}_mses'.format(title)), mses)

        print("============= Config ================", self.config.__dict__)

