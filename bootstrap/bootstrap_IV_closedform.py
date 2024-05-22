import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap, jit, random, scipy, grad, vmap
import haiku as hk
import optax
import chex
import matplotlib.pyplot as plt 

from typing import Tuple
from utils.utils import entropy, Checkpointer, normalize, binary_entropy, clip_norm
from copy import copy
from jax.lax import stop_gradient as sg


class Sampler():
    def __init__(self, 
                 config,
                 discrete=True) -> None:

        # @jax.jit
        def sampler(data):
            x, y, z, beta_p = data
            n, dim = z.shape

            logits = hk.get_parameter("logits", shape=[dim,1], dtype=z.dtype, init=jnp.ones)
            all_probs = jax.nn.softmax(logits, axis=0)
        
            return all_probs, logits

        self.sampler_function = sampler


def get_probs(Z, probs):
    temp = jnp.dot(Z, probs)
    chex.assert_shape(temp, (Z.shape[0], 1))
    return temp


class Model():
    def __init__(self, config) -> None:

        @jax.jit 
        def estimator(data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
                      weights:jnp.ndarray=None,
                      INVERSE_REG=0.0) -> jnp.ndarray:
            
            x, y, z, beta_p = data
            n, _ = y.shape
            _, dx = x.shape
            _, dz = z.shape

            weights = jnp.ones((n, 1)) if weights == None else weights.reshape(n, 1)
 
            ztwz_inv = jnp.linalg.inv((z*weights).T @ z + jnp.eye(dz)*INVERSE_REG)
            chex.assert_shape(ztwz_inv, (dz, dz))

            xtwz = (x*weights).T @ z
            chex.assert_shape(xtwz, (dx, dz))

            xtw_pz_x_inv = jnp.linalg.inv( xtwz @ ztwz_inv @ xtwz.T + jnp.eye(dx)*INVERSE_REG) 
            chex.assert_shape(xtw_pz_x_inv, (dx, dx))

            ztwy = (z*weights).T @ y
            chex.assert_shape(ztwy, (dz, 1))

            xtw_pz_y = xtwz @ ztwz_inv @ ztwy
            chex.assert_shape(xtw_pz_y, (dx, 1))

            param = xtw_pz_x_inv @ xtw_pz_y
            chex.assert_shape(param, (dx,1))


            if config.debug == 1:
                print("=======JIT Check: In 2SLS=======")
                pass

            return param

        self.function = estimator

        @jax.jit 
        def get_inf(param, data, weights, target, INVERSE_REG):
            x, y, z, beta_p = data
            n, _ = y.shape
            _, dx = x.shape
            _, dz = z.shape

            weights = jnp.ones((n, 1)) if weights == None else weights.reshape(n, 1)

            xtwz = (x*weights).T @ z
            chex.assert_shape(xtwz, (dx, dz))
            
            ztwz_inv = jnp.linalg.inv((z*weights).T @ z + jnp.eye(dz)*INVERSE_REG)
            chex.assert_shape(ztwz_inv, (dz, dz))

            xtw_pz_x_inv = jnp.linalg.inv( xtwz @ ztwz_inv @ xtwz.T + jnp.eye(dx)*INVERSE_REG) 
            chex.assert_shape(xtw_pz_x_inv, (dx, dx))

            # Compute the influence function
            y_err = x @ param - y
            chex.assert_shape(y_err, (n, 1))

            ztwy_err = (z*weights) * y_err
            chex.assert_shape(ztwy_err, (n, dz))

            xtw_pz_y_err =  xtwz @ ztwz_inv @ ztwy_err.T
            chex.assert_shape(xtw_pz_y_err, (dx, n))

            param_inf = - xtw_pz_x_inv @ xtw_pz_y_err  
            chex.assert_shape(param_inf, (dx,n))

            chex.assert_shape(target, (dx,1))
            mse_grad = param - target

            inf1 = param_inf.T @ mse_grad
            chex.assert_shape(inf1, (n,1))

            # mse_hessian = Identity
            inf2 =  jnp.sum(param_inf.T * param_inf.T, axis=1, keepdims=True)
            chex.assert_shape(inf2, (n,1))

            influence = inf1  - inf2 #/ n 
            chex.assert_shape(influence, (n,1))

            return influence


        @jax.jit 
        def get_inf_ORACLE(param, data, weights, target, INVERSE_REG):
            x, y, z, beta_p = data
            n, _ = y.shape

            weights = jnp.ones((n, 1)) if weights == None else weights.reshape(n, 1)

            def fn(i):
                w = weights.at[i].set(0)
                return jnp.mean((estimator(data, w, INVERSE_REG) - target)**2)

            LOO_mse = vmap(fn)(jnp.arange(n))
            chex.assert_shape(LOO_mse, (n,))

            mse = jnp.mean((param - target)**2)     
            influence = mse - LOO_mse    
            chex.assert_shape(influence, (n,))

            return influence.reshape((n,1))

        self.get_influence = get_inf
        # self.get_influence = get_inf_ORACLE

    def fit(self, 
            data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], 
            weights:jnp.ndarray=None,
            INVERSE_REG=0.0):
        
        return self.function(data, weights, INVERSE_REG)



class BootstrapMSE():
    def __init__(self, 
                 config, writer) -> None:
        """
        Influence function based bootstrap
        """

        BOOT_SAMPLE_SIZE = config.boot_sample_size
        LAMBDA_REG = config.lambda_reg
        POWER = config.subsampling_power
        LR_START = config.sampler_lr_start
        LR_END = config.sampler_lr_end

        self.config = config
        self.writer=writer
        self.ctr = 0
        self.key = random.PRNGKey(self.config.seed)  # Random state

        # Initialize the distribution function
        sampler_model = Sampler(config)
        sampler =  hk.without_apply_rng(hk.transform(sampler_model.sampler_function))

        # Initialize the distribution function
        beta_model = Sampler(config)
        beta =  hk.without_apply_rng(hk.transform(beta_model.sampler_function))

        twosls_model = Model(config)
        estimator = twosls_model.fit

        opt = optax.adam(learning_rate=optax.linear_schedule(init_value=LR_START, 
                                                                end_value=LR_END,
                                                                transition_steps=self.config.sampler_epochs))

        self.opt = opt
        self.sampler = sampler
        self.beta = beta
        self.estimator = estimator

        # @jax.partial(jax.jit, static_argnames=['k'])
        @jax.jit
        def rejection_sampling(key, probs, beta_avg_probs, data, target):

            beta_p = data[-1]       # The last data index is ASSUMED to have behavior probs
            n, _ = beta_p.shape

            # Ensure that k is even
            # This makes the sample split of equal size
            # This prevents re-jitting between nuisance and theta fit
            k = int(n ** (POWER)) + int(n ** (POWER))%2   

            all_idxs = jnp.arange(n)                # Indices for all of the data
            indices = jnp.ones(k, dtype=int) * -1   # Store the indices selected by rejection sampler
            count = 0

            probs = probs.reshape(-1)
            beta_p = beta_p.reshape(-1)
            beta_avg_probs = beta_avg_probs.reshape(-1)
            chex.assert_equal_shape([probs, beta_p, beta_avg_probs])

            N = config.samples_first_batch + config.samples_per_batch * (config.n_batches - 1)  # total number of samples
            alpha =  (1.0 * n) / N

            pi_eff = alpha * beta_avg_probs + (1-alpha) * probs
            ISratio = pi_eff /  (jax.lax.stop_gradient(beta_avg_probs)) 
            chex.assert_shape(ISratio, (n,))
            
            # Empirical supremum rejection sampling
            ISratio = ISratio / jnp.max(ISratio)# + 1e-6) # Bound the values using normalization
            ISratio = sg(ISratio) 

            key, subkey = jax.random.split(key)
            selected = ISratio > jax.random.uniform(subkey, shape=(n,))

            # Shuffle the order
            # we want random k subsample of the selected items
            # since randomization and selection are order invariant
            # we can first randomize and then select (instead of selecting and then randomizing)
            # By doing this we can stop at the first k selected,
            # instead of scanning the entire list of selected indices.
            # This helps with JITing by not requiring dynamic array sizes.
            key, subkey = jax.random.split(key)
            perm_idxs = jax.random.permutation(subkey, all_idxs)
            selected = selected[perm_idxs]

            # Helper function to find the indices of the first k items
            # that were accepted by the sampler.
            def update_indices(indices_count, i):
                indices, count = indices_count

                # If K items have been selected, no need to update anymore
                # Randomization has been done pre-selction already
                # (instead of randomizing post-selection)
                # Also, Update the number of items selected so far, capped by K
                indices, count = jax.lax.cond(
                    jnp.logical_and(count < k, selected[i]),
                    lambda indices, count: (indices.at[count].set(i), count+1),
                    lambda indices, count: (indices, count),
                    indices, count,
                )

                return (indices, count), None

            # TODO(Enhancement): https://github.com/google/jax/pull/13062
            # Implement early exit of scan once k items have been selected
            indices, count = jax.lax.scan(update_indices, (indices, count), all_idxs)[0]

            weights = indices > -1
            weights = weights * 1.0     # convert to float

            # Indices to unshuffle and select
            undo_perm_idxs = perm_idxs[indices]

            log_dist = jnp.log(pi_eff) 
            log_dist = log_dist[undo_perm_idxs].reshape(k, 1)
            
            data_subset = [item[undo_perm_idxs] for item in data]

            if config.debug == 1:
                print("=======JIT Check: In rejection smapling=======")
                print("Input Shapes: ", ISratio.shape, "+ ...")
    
            param = estimator(data_subset, weights=weights, INVERSE_REG=self.config.inverse_reg)
            influence = twosls_model.get_influence(param, data_subset, weights, target, INVERSE_REG=self.config.inverse_reg)
            return param, influence, weights, log_dist, count

        @jax.jit
        def bootstrap_estimator_mse(params, beta_avg, data, rng):
            """Estimates the MSE of an estimator using statistical bootstrap."""
            
            n, d = data[0].shape

            # Generate bootstrap samples  
            probs = get_probs(Z=data[2], probs=sampler.apply(params=params, data=data)[0])
            chex.assert_shape(probs, (n, 1))

            target = jax.lax.stop_gradient(estimator(data, INVERSE_REG=self.config.inverse_reg))

            def fn(rng):
                return rejection_sampling(rng, probs, beta_avg, data, target)
            
            boot_params, boot_influence, boot_weights, boot_log_dist, boot_counts = vmap(fn)(random.split(rng, BOOT_SAMPLE_SIZE))
            
            boot_params = jax.lax.stop_gradient(boot_params)
            boot_influence = jax.lax.stop_gradient(boot_influence)
            boot_weights = jax.lax.stop_gradient(boot_weights)
            boot_counts = jax.lax.stop_gradient(boot_counts)

            boot_weights = boot_weights.reshape(BOOT_SAMPLE_SIZE, boot_influence.shape[1], 1)

            logp = boot_log_dist  * boot_weights
            chex.assert_shape(logp, (BOOT_SAMPLE_SIZE, boot_influence.shape[1], 1))
            
            loss = jnp.sum(boot_influence * logp * boot_weights) /(jnp.sum(boot_weights) + 1e-10)

            k = int(n ** (POWER)) + int(n ** (POWER))%2   
            # loss = loss +  binary_entropy(probs) * LAMBDA_REG / k     # minimize entropy

            if config.debug == 1:
                print("=======JIT Check: In bootstrap MSE=======")
    
            mse = boot_params - target
            chex.assert_shape(mse, (BOOT_SAMPLE_SIZE, data[0].shape[1], 1))
            mse = jnp.mean(mse**2)

            return loss, (mse, jnp.mean(boot_weights))
        

        @jax.jit
        def update_dist(key, params, beta_avg, opt_state, data):

            grads, aux = grad(bootstrap_estimator_mse, has_aux=True)(params, beta_avg,
                                                    data, 
                                                    key)
            updates, opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, aux

        self.update_dist_fn = update_dist


        @jax.jit
        def beta_loss_fn(params, data, rng):  
            x, y, z, _ = data
            _, beta_logits = beta.apply(params=params, data=data)
            beta_logits = jnp.ones_like(z) * beta_logits.reshape(-1)        # Broadcast to match shape
            chex.assert_equal_shape([beta_logits, z])

            loss = jnp.mean(optax.softmax_cross_entropy(beta_logits, z))
            return loss, loss


        @jax.jit
        def update_beta(key, params, opt_state, data):
            grads, loss = grad(beta_loss_fn, has_aux=True)(params, 
                                                    data, 
                                                    key)
            updates, opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, loss

        self.update_beta_fn = update_beta


    def get_mse(self, data, test_data):
   
        # self.key, subkey = random.split(self.key) 
        estimate = self.estimator(data, INVERSE_REG=self.config.inverse_reg)
        loss = jnp.mean((estimate.reshape(-1) - test_data.reshape(-1))**2)

        if self.config.debug:
            print("True MSE: {}".format(loss))
            print("estimate", estimate)
    
        test_data = jnp.array(test_data)

        # Note: Keep this after evaluation of MSE
        # This internally changes self.key variable
        # Which if not done for oracle and uniform
        # Creates discrepancy in the evlauated MSE between 
        # oracle/uniform and proposed for the FIRST mse.
        if not self.config.algo_name in ['oracle', 'uniform']:
            if self.config.beta_regression:
                self.update_beta(data) 

        return loss, estimate

    def get_beta(self):
        def fn():
            return self.beta.apply(params=self.beta_params, data=self.data)[0] 
        return fn

    def get_sampler(self):
        def fn():
            return self.sampler.apply(params=self.params, data=self.data)[0] 
        return fn
    
    def reset(self, data):
        self.key, subkey = random.split(self.key) 
    
        if self.config.reset_sampler_param:
            self.params = self.sampler.init(rng=subkey, data=data)

        self.key, subkey = random.split(self.key) 
        self.beta_params = self.beta.init(rng=subkey, data=data)

        self.data = data   

    def get_avg_beta(self, data):
        x, y, z, _ = data
        n, xdim = x.shape

        if self.config.beta_regression:
            beta_probs = get_probs(Z=data[2], probs=self.beta.apply(params=self.beta_params, data=data)[0])
            chex.assert_shape(beta_probs, (data[0].shape[0], 1))

            return beta_probs
         
        else:
            raise NotImplementedError
        
    def update_beta(self, data):
        # While doing the optimization it is useful to normalize the data  
        x, y, z, beta_p = [jnp.array(item) for item in data]
        data = (x, y, z, beta_p)           

        # Start the training loop
        losses = []
        checkpointer = Checkpointer(writeout=False)
        checkpointer.save(self.beta_params)
        best_loss = 99999
        ckpt = (self.config.sampler_epochs) // self.config.save_count  

        # Initialize the optimizer
        opt_state = self.opt.init(self.beta_params)
        shift = self.config.sampler_epochs*self.ctr

        for ep in range(1, self.config.sampler_epochs):
            if self.config.algo_name in ['oracle', 'uniform']:
                # Running the baseline
                # No sampler optimization is needed
                return best_loss

            self.key, subkey = random.split(self.key)   
            self.beta_params, opt_state, loss = self.update_beta_fn(subkey, 
                                                            self.beta_params, 
                                                            opt_state, 
                                                            data)
            
            losses.append(loss)

            if ep % ckpt == 0 or ep==self.config.sampler_epochs-1:  
                
                if self.config.debug:
                    self.writer.add_scalar("Bootstrap/Beta loss", loss, shift + ep)

                    probs = self.get_beta()()
                    for i in range(min(probs.shape[0],3)):
                        self.writer.add_scalar("Bootstrap/Beta prob {}".format(i), probs[i], shift + ep)

                avg_loss = np.mean(losses[-ckpt:])
                print("Beta Iterate: ", ep, avg_loss) 

        print("Loss of the beta:", np.mean(losses[-ckpt:]))

        return best_loss


    def update_sampler(self, data, Dx):   
        self.ctr += 1

        # self.update_beta(data) #Running in get_mse instead

        x, y, z, beta_p = [jnp.array(item) for item in data]
        data = (x, y, z, beta_p)           

        # Start the training loop
        losses = []
        checkpointer = Checkpointer(writeout=False)
        checkpointer.save(self.params)
        best_loss = 99999
        ckpt = (self.config.sampler_epochs) // self.config.save_count

        # Initialize the optimizer
        opt_state = self.opt.init(self.params)


        for ep in range(1, self.config.sampler_epochs):
            if self.config.algo_name in ['oracle', 'uniform']:
                # Running the baseline
                # No sampler optimization is needed
                return best_loss

            beta_avg = self.get_avg_beta(data)
            self.key, subkey = random.split(self.key)

            self.params, opt_state, aux = self.update_dist_fn(subkey, 
                                                            self.params, 
                                                            beta_avg,
                                                            opt_state, 
                                                            data)
            loss = aux[0]
            losses.append(loss)


            if ep % ckpt == 0  or ep==self.config.sampler_epochs-1: 
                if self.config.debug:        
                    shift = self.config.sampler_epochs*(self.ctr-1)
                    self.writer.add_scalar("Bootstrap/proxy mse", loss, shift + ep)
                    self.writer.add_scalar("Bootstrap/valid subsample fraction", aux[1], shift + ep)

                    probs = self.get_sampler()()
                    for i in range(min(probs.shape[0],3)):
                        self.writer.add_scalar("Bootstrap/prob {}".format(i), probs[i], shift + ep)

                avg_loss = np.mean(losses[-ckpt:])
                print("Iterate: ", ep, avg_loss) 

        print("Loss of the estimator:", np.mean(losses[-ckpt:]))

        return best_loss



