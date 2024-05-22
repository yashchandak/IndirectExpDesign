import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap, jit, random, grad
import haiku as hk
import optax
import chex
from typing import Tuple
from utils.utils import Checkpointer,  split_data, breakpoint_if_nonfinite, clip_norm
from copy import copy
import os


class Model:
    def __init__(self, config, writer=None) -> None:
        self.config = config
        self.writer=writer
        LR_START = config.lr_start
        LR_END = config.lr_end
        INNER_EPOCHS = config.inner_epochs
        REGULARIZER = config.inverse_reg

        bce =  optax.sigmoid_binary_cross_entropy
        # sigmoid = stable_sigmoid 
        sigmoid = jax.nn.sigmoid 


        ####################################
        # Nuisance functions / losses
        ####################################

        # @jax.jit
        def nuisance_fwd(data):
            X, Z, T, Y, beta_p = data
            n, xdim = X.shape
            _, zdim = Z.shape 

            # f = jax.nn.sigmoid
            f = lambda x: x #identity

            if config.env_name == 'synthetic':
                h2 = jnp.hstack([X, Z])
                chex.assert_shape(h2, (n, xdim+zdim))
                # h2 = Z
            else:
                DIM = 8
                # Shared representation of X for all the nuisance function
                # h = jax.nn.sigmoid(hk.Linear(DIM)(X)) 
                h2 = jnp.hstack([X, Z]) 
                h2 = jax.nn.sigmoid(hk.Linear(DIM)(h2))
                chex.assert_shape(h2, (n, DIM))#TODO
                # chex.assert_shape(h2, (n, xdim+zdim))

            T_ZX = f(hk.Linear(1, with_bias=True)(h2))

            # chex.assert_shape(h2, (n, DIM+zdim))
            # chex.assert_shape(h2, (n, zdim))
            if config.debug == 1:
                print("=======JIT Check: In Nuisance FWD Model=======")
                print("Input Shapes: ", X.shape, Z.shape, T.shape, Y.shape, beta_p.shape)

            return T_ZX

        nuisance_net = hk.without_apply_rng(hk.transform(nuisance_fwd))
        nuisance_opt = optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=LR_START, end_value=LR_END, transition_steps=INNER_EPOCHS
            ),
            weight_decay=REGULARIZER
        )
        nuisance_opt_outer = optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=config.estimator_lr_start, 
                end_value=config.estimator_lr_end,
                transition_steps=self.config.epochs
            ),
            weight_decay=REGULARIZER
        )

        self.nuisance_net = nuisance_net
        self.nuisance_opt = nuisance_opt    
        self.nuisance_opt_outer = nuisance_opt_outer    

        @jax.jit
        def nuisance_loss_fn(
            nuisance_params,
            data,
            weights: jnp.ndarray = None,
            key=None,
        ):
            """
            Loss function for all the nuisance parameters
            """
            X, Z, T, Y, beta_p = data
            n, _ = Y.shape

            weights = jnp.ones((n, 1)) if weights == None else weights.reshape(n, 1)

            T_ZX = nuisance_net.apply(nuisance_params, data)
            losses = bce(T_ZX, T) 
            
            chex.assert_shape(losses, (n, 1))

            losses = losses * weights 
            chex.assert_shape(losses, (n, 1))

            loss = jnp.sum(losses) / (jnp.sum(weights) + 1e-10) #TODO setting loss to zero instantly results in NaN

            if config.debug == 1:
                print("=======JIT Check: In Nuisance Loss =======")
                print("Input Shapes: ", "...")

            return loss, loss
        

        @jax.jit
        def nuisance_loss_fn_datawrapper(
            nuisance_params,
            data,
            pos:jnp.ndarray = None,
            weights: jnp.ndarray = None):

            data_subset = [item[pos] for item in data]
            return nuisance_loss_fn(nuisance_params, data_subset, weights)

        self.nuisance_loss_fn_datawrapper = nuisance_loss_fn_datawrapper

        @jax.jit
        def nuisance_update_fn(
            nuisance_params, nuisance_opt_state, data, weights=None
            ):

            grads, loss = grad(nuisance_loss_fn, has_aux=True)(
                nuisance_params, data, weights
            )
            updates, nuisance_opt_state = nuisance_opt.update(grads, 
                                                              nuisance_opt_state,
                                                              params=nuisance_params)
            new_nuisance_params = optax.apply_updates(nuisance_params, updates)

            return new_nuisance_params, nuisance_opt_state, loss


        @jax.jit
        def nuisance_update_fn_outer(
            nuisance_params, nuisance_opt_state, data, weights=None
        ):
            grads, loss = grad(nuisance_loss_fn, has_aux=True)(
                nuisance_params, data, weights
            )
            updates, nuisance_opt_state = nuisance_opt_outer.update(grads, 
                                                              nuisance_opt_state,
                                                              params=nuisance_params)
            new_nuisance_params = optax.apply_updates(nuisance_params, updates)

            return new_nuisance_params, nuisance_opt_state, loss

        self.nuisance_update_fn = nuisance_update_fn
        self.nuisance_update_fn_outer = nuisance_update_fn_outer

        ####################################
        # CATE functions / losses
        ####################################

        # @jax.jit
        def fwd(X, T):
            n, xdim = X.shape

            if config.env_name == 'synthetic':
                h2 = jnp.hstack([X, T])
                chex.assert_shape(h2, (n, xdim+1))
            else:
                DIM = 8
                # h = jax.nn.sigmoid(hk.Linear(DIM)(X)) 
                h2 = jnp.hstack([X, T])   
                h2 = jax.nn.sigmoid(hk.Linear(DIM)(h2))
                chex.assert_shape(h2, (n, DIM)) 
                # chex.assert_shape(h2, (n, xdim+1)) 

            Y = hk.Linear(1, with_bias=True)(h2)

            chex.assert_shape(Y, (n, 1))

            if config.debug == 1:
                print("=======JIT Check: In Estimator FWD Model=======")
                print("Input Shapes: ", X.shape)

            return Y


        # Using adam without bias regularization
        net = hk.without_apply_rng(hk.transform(fwd))
        opt = optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=LR_START, end_value=LR_END, transition_steps=INNER_EPOCHS
            ),
            weight_decay=REGULARIZER
        )
        opt_outer = optax.adamw(
            learning_rate=optax.linear_schedule(
                init_value=config.estimator_lr_start, 
                end_value=config.estimator_lr_end,
                transition_steps=self.config.epochs
            ),
            weight_decay=REGULARIZER
        )

        self.net = net
        self.opt = opt      
        self.opt_outer = opt_outer      

        @jax.jit
        def loss_fn(
            params,
            nuisance_params,
            data,
            weights: jnp.ndarray = None
        ):
            """
            DeepIV [Hartford et al. 2017]
            """
            X, Z, T, Y, beta_p = data
            n, _ = Y.shape

            weights = jnp.ones((n, 1)) if weights == None else weights.reshape(n, 1)

            T_ZX = nuisance_net.apply(nuisance_params, data)
            T_ZX = sigmoid(T_ZX) 
            
            chex.assert_shape(T_ZX, (n, 1))

            ones = jnp.ones_like(T_ZX)
            y_0 = net.apply(params, X, 1 - ones)
            y_1 = net.apply(params, X, ones)
            y_hat = y_1 * T_ZX + y_0 * (1 - T_ZX)  # Expectation under binary distribution # DeepIV loss
            chex.assert_shape(y_hat, (n,1))

            losses = (y_hat - Y) ** 2   
            chex.assert_shape(losses, (n, 1))

            losses = losses * weights
            loss = jnp.sum(losses) / (jnp.sum(weights) + 1e-10) 

            if config.debug == 1:
                print("=======JIT Check: In Estimator Loss =======")
                print("Input Shapes: ", "...")

            return loss, (loss, -999.0)
        

        @jax.jit
        def loss_fn_datawrapper(
            params,
            nuisance_params,
            data,
            pos:jnp.ndarray = None,
            weights: jnp.ndarray = None):

            data_subset = [item[pos] for item in data]
            return loss_fn(params, nuisance_params, data_subset, weights)

        self.loss_fn_datawrapper = loss_fn_datawrapper

        @jax.jit
        def update_fn(
            params, nuisance_params, opt_state, data, weights=None
        ):
            grads, aux = grad(loss_fn, has_aux=True)(
                params, nuisance_params, data, weights,
            )
            
            updates, opt_state = opt.update(grads, opt_state, params=params)
            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, aux

        @jax.jit
        def update_fn_outer(
            params, nuisance_params, opt_state, data, weights=None
        ):
            grads, aux = grad(loss_fn, has_aux=True)(
                params, nuisance_params, data, weights
            )
            
            updates, opt_state = opt_outer.update(grads, opt_state, params=params)
            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, aux

        self.update_fn = update_fn
        self.update_fn_outer = update_fn_outer

        ####################################
        # Optimization
        ####################################

        @jax.jit
        def fit_jit(
            key, data, params=None, nuisance_params=None, weights=None
        ):
            """
            This fit function is called during inner optimization of the model.
            This is supposed to be cheaper and faster.

            Since this will be called from within bootstrap with subsampled dataset
            We do not split the dataset further. Both nuisance and theta use the same
            set of data.
            """

            opt_state = opt.init(params)
            nuisance_opt_state = nuisance_opt.init(nuisance_params)

            def nuisance_fn(carry, idx):
                nuisance_params, nuisance_opt_state = carry
                nuisance_params, nuisance_opt_state, loss = nuisance_update_fn(
                    nuisance_params=nuisance_params,
                    nuisance_opt_state=nuisance_opt_state,
                    data=data, 
                    weights=weights
                )
                return (nuisance_params, nuisance_opt_state), loss

            nuisance_params, nuisance_opt_state = jax.lax.scan(
                nuisance_fn,
                (nuisance_params, nuisance_opt_state),
                jnp.arange(config.inner_epochs),
            )[0]

            def fn(carry, idx):
                params, opt_state = carry
                params, opt_state, aux = update_fn(
                    params=params,
                    nuisance_params=nuisance_params, 
                    opt_state=opt_state, 
                    data=data, 
                    weights=weights
                )
                # key, params, nuisance_params, opt_state, data, weights=None, mask=None
                return (params, opt_state), aux[0]

            params, opt_state = jax.lax.scan(
                fn, (params, opt_state), jnp.arange(config.inner_epochs)
            )[0]
            
            if config.debug == 1:
                print("=======JIT Check: In FIT JIT =======")
                print("Input Shapes: ", "...")

            return params, nuisance_params

        self.fit_jit = fit_jit

    def fit(
        self,
        key,
        data,
        ctr=1
    ):
        """
        This fit function is called to train the model once the seach for sampling policy is over.
        This is more reliable and slightly more expensive.
        """

        # data_1, data_2 = split_data(data) #TODO
        data_1, data_2 = data, data

        #########################################################
        # Train the Nuisance parameters
        #########################################################
        key, subkey = jax.random.split(key)
        self.nuisance_params = self.nuisance_net.init(subkey, data_1)

        losses = []
        checkpointer = Checkpointer(writeout=False)
        checkpointer.save(self.nuisance_params)
        best_loss = 99999
        T = self.config.epochs // self.config.save_count

        # Initialize the optimizer
        nuisance_opt_state = self.nuisance_opt_outer.init(self.nuisance_params)

        for ep in range(1, self.config.epochs):
            self.nuisance_params, nuisance_opt_state, loss = self.nuisance_update_fn_outer(
                nuisance_params=self.nuisance_params,
                nuisance_opt_state=nuisance_opt_state,
                data=data_1
            )

            losses.append(loss)

            if ep % T == 0 or ep==self.config.epochs-1:                
                if self.config.debug and self.writer is not None:
                    shift = (self.config.epochs)*ctr
                    self.writer.add_scalar("Estimator/nuisance loss", losses[-1], shift + ep)

                avg_loss = np.mean(losses[-T:])
                print("Nuisance Iterate: ", ep, avg_loss)

        print("Loss of the Nuisance MODEL:", np.mean(losses[-T:]))

        #########################################################
        # Train the main parameters
        #########################################################
        key, subkey = jax.random.split(key)
        self.est_params = self.net.init(subkey, data_2[0], data_2[2])

        losses = []
        checkpointer = Checkpointer(writeout=False)
        checkpointer.save(self.est_params)
        best_loss = 99999
        T = self.config.epochs // self.config.save_count

        # Initialize the optimizer
        opt_state = self.opt_outer.init(self.est_params)

        for ep in range(1, self.config.epochs):
            self.est_params, opt_state, aux = self.update_fn_outer(
                params=self.est_params,
                nuisance_params=self.nuisance_params, 
                opt_state=opt_state, 
                data=data_2
            )

            losses.append(aux[0])

            if ep % T == 0 or ep==self.config.epochs-1:                
                if self.config.debug and self.writer is not None:
                    shift = (self.config.epochs)*ctr
                    self.writer.add_scalar("Estimator/main loss", losses[-1], shift + ep)
                    
                avg_loss = np.mean(losses[-T:])
                print("Estimator Iterate: ", ep, avg_loss, aux[1])

        print("Loss of the Estimator:", np.mean(losses[-T:]))
        return self.est_params
