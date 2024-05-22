import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap, jit, random, scipy, grad, vmap
from functools import partial
import haiku as hk
import optax
import chex
import matplotlib.pyplot as plt
from jax.experimental import checkify
import os
from estimators.CIV_discrete import Model
from typing import Tuple
from utils.utils import entropy, binary_entropy, Checkpointer, normalize, tree_dot, tree_add, split_data, breakpoint_if_nonfinite, clip_norm, conjugate_gradient_hvp, neumann_hvp
from copy import copy
from jax.lax import stop_gradient as sg

class Sampler:
    def __init__(self, config) -> None:
        """
        Sampler function for _binary_ instruments conditioned on the covariates
        """
        # @jax.jit
        def sampler(X):
            n, _ = X.shape 
            h = X  
            h = jax.nn.sigmoid(hk.Linear(8)(h)) 
            logits = hk.Linear(config.n_IV)(h)
            all_probs = jax.nn.softmax(logits, axis=-1)
            chex.assert_shape(all_probs, (n, config.n_IV))

            if config.debug == 1:                
                print("=======JIT Check: In Sampler FWD=======")
                print("Input Shape: ", X.shape)

            return all_probs, logits

        self.sampler_function = sampler


def get_probs(Z, probs):
    """Assumes Z is one-hot. No check done for that."""
    chex.assert_equal_shape([Z, probs])
    return jnp.sum(Z * probs, axis=-1, keepdims=True)      # For arbitrary number of IV


class BootstrapMSE:
    def __init__(self, config, writer) -> None:
        """
        Rejection sampling Bootstrap MSE
        with Influence function based derivatives
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

        # Initialize the sampling function
        sampler_model = Sampler(config)
        sampler = hk.without_apply_rng(hk.transform(sampler_model.sampler_function))

        # Initialize the distribution function
        beta_model = Sampler(config)
        beta =  hk.without_apply_rng(hk.transform(beta_model.sampler_function))

        # Initialize the estimator
        est_model = Model(config, writer)
        estimator = est_model.fit_jit

        # Optimizer for sampler
        opt = optax.adam(
            learning_rate=optax.linear_schedule(
                init_value=LR_START,
                end_value=LR_END,
                transition_steps=self.config.sampler_epochs,
            )
        )

        self.opt = opt
        self.sampler = sampler
        self.beta = beta
        self.est_model = est_model
        self.estimator = estimator
        self.all_sampler_params = []

        @jax.jit
        def rejection_sampling(key, probs, beta_avg_probs, data, est_params, nuisance_params):
            """
            Selects k out of n samples, without replacement
            Using acceptance/rejection sampling to simulate a different distribution
            """

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
            ISratio = pi_eff /  beta_avg_probs 
            chex.assert_shape(ISratio, (n,))
            
            # Empirical supremum rejection sampling
            ISratio = ISratio / jnp.max(ISratio)# + 1e-6) # Bound the values using normalization
            ISratio = sg(ISratio) 
            
            # Probs correspond to the probability of selecting 1
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

            log_dist = jnp.log(pi_eff) # pi_eff will never go to zero + 1e-10) # Check the impact of eps here
            log_dist = log_dist[undo_perm_idxs].reshape(k, 1)

            data_subset = [item[undo_perm_idxs] for item in data]

            if config.debug == 1:
                print("=======JIT Check: In rejection smapling=======")
                print("Input Shapes: ", ISratio.shape, "+ ...")

            est_params, nuisance_params = estimator(
                key=key,
                data=data_subset,
                params=est_params,
                nuisance_params=nuisance_params,
                weights=weights,
            )
            return est_params, nuisance_params, undo_perm_idxs, weights, log_dist, count


        @jax.jit
        def bootstrap_estimator_mse(
            sampler_params, est_params_set, nuisance_params_set, data, beta_avg, Dx, Dt, y_preds, key
        ):
            """Estimates the MSE of an estimator using subsampling bootstrap."""

            n, _ = data[0].shape
            nd, _ = Dx.shape
            key, subkey = random.split(key)

            # Get the probabilities of data samples under the learned sampler
            all_probs = sampler.apply(params=sampler_params, X=data[0])[0]
            probs = get_probs(Z=data[1], probs=all_probs)
            chex.assert_shape(probs, (n, 1))

            beta_avg = sg(beta_avg)
            chex.assert_shape(beta_avg, (n, 1))

            # Generate bootstrap samples
            def fn(est_params, nuisance_params, key):
                return rejection_sampling(
                    key=key, probs=probs, beta_avg_probs=beta_avg, data=data, est_params=est_params, nuisance_params=nuisance_params
                )

            # Find the parmaters for different bootstraped estimators
            (
                boot_est_params,
                boot_nuisance_params,
                boot_pos,
                boot_weights,
                boot_log_dist,
                boot_counts,
            ) = vmap(fn)(est_params_set, nuisance_params_set, random.split(subkey, BOOT_SAMPLE_SIZE))  
            chex.assert_shape(boot_counts, (BOOT_SAMPLE_SIZE,))

            # These stop grads (particularly, the first two) are VERY IMPORTANT
            boot_est_params = sg(boot_est_params)
            boot_nuisance_params = sg(boot_nuisance_params)
            boot_pos = sg(boot_pos)
            boot_weights = sg(boot_weights)
            boot_counts = sg(boot_counts)
            # Do NOT stop grad boot_log_dist. That has the required path to sampler.

            # Given the learned parameters for the different bootstraped estimators
            # Now compute the (proxy) mse using these
            @jax.jit
            def proxy_mse(param):    
                """
                Compute (proxy) MSE for each learned estimator, and then average over all 
                """
                preds = est_model.net.apply(param, Dx, Dt)
                chex.assert_shape(preds, (nd, 1))

                error = preds - y_preds.reshape(nd, 1)
                chex.assert_shape(error, (nd, 1))

                # Compute mse
                mse = jnp.mean(error**2)  
                
                if config.debug == 1:
                    print("=======JIT Check: In Proxy MSE=======")
                    print("Input Shapes: ", " ...")

                return mse, mse

            mse_grads, mses = vmap(jax.grad(proxy_mse, has_aux=True))(boot_est_params)
            mse = jnp.mean(mses)

            mse_grads = sg(mse_grads) #TODO

            # TODO(Enhancement): Use donate_argnums to cache unchanged branch
            # https://github.com/google/jax/discussions/6237
            @jax.jit
            def get_influence(boot_est_par,
                                boot_nuisance_par,
                                boot_po,
                                boot_weigh,
                                boot_coun,
                                mse_gra):
                
                def element_wise(idx):
                    """
                    Compute influence of each sample
                    """
                    ######################################
                    # Influence of nuisance functions
                    #####################################

                    def jvp_fn_nuisance(param, target):
                        """
                        Helper function to copmute Jacobian vector product
                        """
                        def gradv_nuisance(param):
                            # TODO(Enhancement): The folowing grad copmutation is redundant
                            # In pytorch one could have saved the compute graph and reused it
                            # Not sure how to do the same in JAX.
                            #
                            # Note: Hessian needs to be computed on the entire data
                            grad, _ = jax.grad(est_model.nuisance_loss_fn_datawrapper, has_aux=True)(
                                param,
                                data=data,
                                pos=boot_po,        
                                weights=boot_weigh
                            )
                            return tree_dot(grad, target)

                        Jv = jax.grad(gradv_nuisance)(param)
                        # Add Hessian component of L2 regularizer
                        # as grad of L2 reg is not explicit through the loss, adamW computes that internally.
                        Jv_reg = tree_add(Jv, target, scale_rhs=self.config.inverse_reg)  #(H +\lambda I)v = Hv + \lambda v
                        return Jv_reg
                    
                    HVP_nuisance = lambda v: jvp_fn_nuisance(boot_nuisance_par, v)

                    q, _ = jax.grad(est_model.nuisance_loss_fn_datawrapper, has_aux=True)(
                        boot_nuisance_par,
                        data=data,
                        pos=boot_po[idx],
                        weights=boot_weigh[idx]
                    )
                    
                    q = sg(q)
                    if config.hess_inv == 'CG': 
                        H_inv_q = conjugate_gradient_hvp(HVP_nuisance, b=q, max_iter=config.hinv_iter)
                    elif config.hess_inv == 'Neumann':
                        H_inv_q = neumann_hvp(HVP_nuisance, b=q, max_iter=config.hinv_iter, alpha=config.neumann_alpha, debug=config.debug)
                    else: raise ValueError("Unknown function for inverse computation")                    
                    H_inv_q = sg(H_inv_q)                                # H_{nuisance}^{-1}q
            
                    def jvp_fn_Jac(est_param, nuisance_param, target):
                        def gradv3(est_param, nuisance_param):
                            # Differentiate wrt to nuisance params (not with respect to est_params)
                            grad, _ = jax.grad(est_model.loss_fn_datawrapper, has_aux=True, argnums=1)(
                                est_param,
                                nuisance_param,
                                data=data,
                                pos=boot_po,        
                                weights=boot_weigh,
                            )
                            return tree_dot(grad, target)       # <\nabla f, v> 
                        
                        # Differentiate wrt to est params 
                        return jax.grad(gradv3)(est_param, nuisance_param)           # \nabla <\nabla f, v> = <\nabla^2 f , v>

                    J_H_inv_q = jvp_fn_Jac(boot_est_par, boot_nuisance_par, H_inv_q)        # J_{main, nuisance}  H_{nuisance}^{-1}q
                    J_H_inv_q = sg(J_H_inv_q)

                    ######################################
                    # Influence of CATE functions
                    #####################################

                    def jvp_fn(param, target):
                        """
                        Helper function to copmute Jacobian vector product
                        """
                        def gradv(param):
                            # TODO(Enhancement): The folowing grad copmutation is redundant
                            # In pytorch one could have saved the compute graph and reused it
                            # Not sure how to do the same in JAX.
                            #
                            # Note: Hessian needs to be computed on the entire data
                            grad, _ = jax.grad(est_model.loss_fn_datawrapper, has_aux=True)(
                                param,
                                nuisance_params=boot_nuisance_par,
                                data=data,
                                pos=boot_po,        
                                weights=boot_weigh,
                            )
                            return tree_dot(grad, target)

                        Jv = jax.grad(gradv)(param)
                        # Add Hessian component of L2 regularizer
                        # as grad of L2 reg is not explicit through the loss, adamW computes that internally.
                        Jv_reg = tree_add(Jv, target, scale_rhs=self.config.inverse_reg)  #(H +\lambda I)v = Hv + \lambda v
                        return Jv_reg

                    HVP_cate = lambda v: jvp_fn(boot_est_par, v)                    
                    m, _ = jax.grad(est_model.loss_fn_datawrapper, has_aux=True)(
                        boot_est_par,
                        nuisance_params=boot_nuisance_par,
                        data=data,
                        pos=boot_po[idx],
                        weights=boot_weigh[idx],
                    )
                    
                    m = sg(m)
                    m = tree_add(m, J_H_inv_q, scale_rhs=-1)  # v = v - nuisance_p
    
                    if config.hess_inv == 'CG': 
                        inf_theta = conjugate_gradient_hvp(HVP_cate, b=m, max_iter=config.hinv_iter)
                    elif config.hess_inv == 'Neumann':
                        inf_theta = neumann_hvp(HVP_cate, b=m, max_iter=config.hinv_iter, alpha=config.neumann_alpha, debug=config.debug)
                    else: raise ValueError("Unknown function for inverse computation")                    
                    inf_theta = sg(inf_theta)                                # H_{nuisance}^{-1}q
            
                    if config.debug == 1:
                        print("=======JIT Check: In Left-Hessian-Inv_right =======")
                        print("Input Shapes: ", " ...")

                    ########################
                    ## First order influence
                    ##########################

                    inf1 = - tree_dot(inf_theta, sg(mse_gra))                                    # - v^T H^{-1} g

                    ##########################
                    ## Second order influence (partial)
                    ##########################

                    def jvp_fn_mse(param, target):
                        def gradv2(param):
                            grad, _ = jax.grad(proxy_mse, has_aux=True)(param)      
                            return tree_dot(grad, target)       # <\nabla f, v> 
                        return jax.grad(gradv2)(param)           # \nabla <\nabla f, v> = <\nabla^2 f , v>

                    inf2 = tree_dot(inf_theta, jvp_fn_mse(boot_est_par, inf_theta))        # g H^{-1} h H^{-1}g
                    inf2 = inf2 / boot_coun    # weight them absolutely by eps and eps^2 or  relatively by 1 and eps.

                    ###########################

                    inf = sg(inf1 - inf2) 
                    return inf, inf1     

                data_inf, data_inf1 = vmap(element_wise)(jnp.arange(boot_po.shape[0]).reshape(-1, 1)) 


                if config.debug > 0:
                    pass #TODO

                return data_inf

            checked_lihr = checkify.checkify(get_influence)
            err, mse_influence = vmap(checked_lihr)(boot_est_params,
                                                    boot_nuisance_params,
                                                    boot_pos,
                                                    boot_weights,
                                                    boot_counts,
                                                    mse_grads

            )
            chex.assert_shape(mse_influence, (BOOT_SAMPLE_SIZE, boot_pos.shape[1])) 


            mse_influence = sg(mse_influence.reshape(BOOT_SAMPLE_SIZE, boot_pos.shape[1], 1))
            boot_weights = boot_weights.reshape(BOOT_SAMPLE_SIZE, boot_pos.shape[1], 1)

            logp = boot_log_dist  * boot_weights
            chex.assert_shape(logp, (BOOT_SAMPLE_SIZE, boot_pos.shape[1], 1))

            loss = jnp.sum(mse_influence * logp * boot_weights) /(jnp.sum(boot_weights) + 1e-10)

            # loss = loss + LAMBDA_REG * entropy(all_probs)  # Minimize entropy #TODO Entropy of effective policy vs entropy of policy for next deployment

            if config.debug == 1:
                print("=======JIT Check: In bootstrap MSE=======")
                print("Input Shapes: ", " ...")

                breakpoint_if_nonfinite(jnp.mean(mse_influence))
                # breakpoint_if_nonfinite(loss)
                # breakpoint_if_nonfinite(mse)

            return loss, (mse, 
                          boot_est_params, 
                          boot_nuisance_params, 
                          loss,
                          jnp.mean(probs),
                          jnp.mean(boot_pos),
                          jnp.mean(boot_weights),
                          jnp.mean(mse_influence),
                          jnp.mean(logp),
                          err)




        @jax.jit
        def update_dist(
            key,
            sampler_params,
            est_params_set,
            nuisance_params_set,
            opt_state,
            data,
            beta_avg,
            Dx,
            Dt,
            y_preds,
        ):
            grads, aux = grad(bootstrap_estimator_mse, has_aux=True)(
                sampler_params, est_params_set, nuisance_params_set, data, beta_avg, Dx, Dt, y_preds, key
            )

            updates, opt_state = opt.update(grads, opt_state)#, params=sampler_params)
            new_params = optax.apply_updates(sampler_params, updates)

            return new_params, opt_state, (*aux, jnp.sqrt(tree_dot(grads, grads)))

        self.update_dist_fn = update_dist

        @jax.jit
        def beta_loss_fn(params, data, rng):  
            x, z, _, _, _ = data
            _, beta_logits = beta.apply(params=params, X=x)
            # beta_logits = jnp.ones_like(z) * beta_logits.reshape(-1)        # Broadcast to match shape
            chex.assert_equal_shape([beta_logits, z])

            loss = jnp.mean(optax.softmax_cross_entropy(beta_logits, z))
            return loss, loss


        @jax.jit
        def update_beta_dist(key, params, opt_state, data):
            grads, loss = grad(beta_loss_fn, has_aux=True)(params, data, key)
            updates, opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, loss

        self.update_beta_fn = update_beta_dist
        
    def get_mse(self, train_data, test_data):
        
        train_data = [jnp.array(item) for item in train_data]
        
        self.key, subkey = random.split(self.key)
        est_params = self.est_model.fit(subkey, train_data, self.ctr)

        # Here we are assuming that Dx available during training is the same as in the test data
        # From here we also cache y_preds which is later used as the 
        # "centering" value for the bootstrap
        (Dx, Dt), Dy = test_data
        self.y_preds = self.est_model.net.apply(est_params, Dx, Dt)
        chex.assert_equal_shape([Dy, self.y_preds])

        losses = (Dy - self.y_preds) ** 2
        chex.assert_shape(losses, (Dy.shape[0], 1))

        if self.config.debug:
            print("True MSE: {}".format(jnp.mean(losses)))

        # Note: Keep this after evaluation of MSE
        # This internally changes self.key variable
        # Which if not done for oracle and uniform
        # Creates discrepancy in the evlauated MSE between 
        # oracle/uniform and proposed for the FIRST mse.
        if not self.config.algo_name in ['oracle', 'uniform']:
            if self.config.beta_regression:
                self.update_beta(train_data)     

        return jnp.mean(losses), self.y_preds

    def get_beta(self):
        def fn(X):
            return self.beta.apply(params=self.beta_params, X=X)[0] 
        return fn
    
    def get_sampler(self):
        def fn(X):
            return self.sampler.apply(params=self.sampler_params, X=X)[0]  
        return fn

    def inner_reset(self, data):
        self.key, subkey = random.split(self.key)
        self.est_params_set = vmap(self.est_model.net.init, in_axes=(0,None,None))(
                                random.split(subkey, self.config.boot_sample_size), data[0], data[2])
        
        assert all(param.shape[0] == self.config.boot_sample_size
               for param in jax.tree_util.tree_leaves(self.est_params_set))
    
        self.key, subkey = random.split(self.key)
        self.nuisance_params_set = vmap(self.est_model.nuisance_net.init, in_axes=(0,None))(
                                    random.split(subkey, self.config.boot_sample_size), data)
        
        assert all(param.shape[0] == self.config.boot_sample_size
               for param in jax.tree_util.tree_leaves(self.nuisance_params_set))


    def reset(self, data):
        self.key, subkey = random.split(self.key)
        if self.config.reset_sampler_param:
           self.sampler_params = self.sampler.init(rng=subkey, X=data[0])

        self.key, subkey = random.split(self.key) 
        self.beta_params = self.beta.init(rng=subkey, X=data[0])

        self.inner_reset(data)
        self.data = data 

    def get_avg_beta(self, data):
        x, z, _, _, _ = data
        n, xdim = x.shape

        if self.config.beta_regression:
            return get_probs(probs=self.beta.apply(params=self.beta_params, X=x)[0], Z=z) 
         
        else:
            weights = np.zeros((1, self.ctr))
            probs = np.zeros((n, self.ctr))

            for idx in range(self.ctr):
                if idx == 0:
                    weights[0, idx] = self.config.samples_first_batch
                    probs[:, idx] = 1.0/ self.config.n_IV       # First batch is uniform distirbution over all IVs. 
                    # uniform distirbution
                else:
                    weights[0, idx] = self.config.samples_per_batch
                    param = self.all_sampler_params[idx-1]
                    probs[:, idx] = get_probs(probs=self.beta.apply(params=param, X=x)[0], Z=z).reshape(-1)                 

            # Compute the weighted mean of the probabilities
            weights = weights / jnp.sum(weights)       
            probs = probs * weights
            probs = np.sum(probs, axis=-1, keepdims=True)   
            chex.assert_shape(probs, (n,1))      

            return jnp.array(probs)


    def update_beta(self, data):
        # While doing the optimization it is useful to normalize the data  
        x, z, a, r, beta_p = [jnp.array(item) for item in data]
        data = (x, z, a, r, beta_p)           

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
            if self.config.algo_name in ['uniform']:
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
                avg_loss = np.mean(losses[-ckpt:])
                print("Beta Iterate: ", ep, avg_loss)   
                    
        print("Loss of the beta:", np.mean(losses[-ckpt:]))

        return best_loss


    def update_sampler(self, data, Dx_t):
        Dx, Dt = Dx_t
        self.ctr += 1

        X, Z, T, Y, beta_p = [jnp.array(item) for item in data] 
        data = (X, Z, T, Y, beta_p)

        # Start the training loop
        losses = []
        checkpointer = Checkpointer(writeout=False)
        checkpointer.save(self.sampler_params)
        best_loss = 99999
        ckpt = (self.config.sampler_epochs) // self.config.save_count

        # Initialize the optimizer
        opt_state = self.opt.init(self.sampler_params)

        for ep in range(1, self.config.sampler_epochs):
            if self.config.algo_name in ['uniform']:
                # Running the baseline
                # No sampler optimization is needed
                return best_loss

            beta_avg = self.get_avg_beta(data)
            self.key, subkey = random.split(self.key)

            self.sampler_params, opt_state, aux = self.update_dist_fn(
                subkey,
                self.sampler_params,
                self.est_params_set,
                self.nuisance_params_set,
                opt_state,
                data,
                beta_avg,
                Dx,
                Dt,
                self.y_preds,
            )

            loss, self.est_params_set, self.nuisance_params_set, *more_aux = aux
            losses.append(loss)

            err = more_aux[-2]
            err.throw()

            if self.config.reset_inner_param: self.inner_reset(self.data)
                
            if ep % ckpt == 0 or ep==self.config.sampler_epochs-1:
                if self.config.debug:
                    shift = self.config.sampler_epochs*(self.ctr-1)
                    self.writer.add_scalar("Bootstrap/proxy mse", loss, shift + ep)
                    self.writer.add_scalar("Bootstrap/Loss-grad mix", more_aux[0], shift + ep)
                    self.writer.add_scalar("Bootstrap/Prob of old samples under new dist", more_aux[1], shift + ep)
                    self.writer.add_scalar("Bootstrap/Avg Pos of selected subsamples", more_aux[2], shift + ep)
                    self.writer.add_scalar("Bootstrap/Valid fraction within subsamples", more_aux[3], shift + ep)
                    self.writer.add_scalar("Bootstrap/Mean Influence", more_aux[4], shift + ep)
                    self.writer.add_scalar("Bootstrap/Mean log_p", more_aux[5], shift + ep)
                    self.writer.add_scalar("Bootstrap/grad norm-var", more_aux[-1], shift + ep)

                avg_loss = np.mean(losses[-ckpt:])
                print("Sampler Iterate: ", ep, avg_loss)

        self.all_sampler_params.append(self.sampler_params)

        print("Loss of the Sampler:", np.mean(losses[-ckpt:]))

        return best_loss #np.mean(losses)