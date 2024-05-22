import jax
import pickle
import jax.numpy as jnp
from pathlib import Path
import copy 
import os
import sys
import numpy as np

import shutil
from time import time
from os import path, mkdir, listdir, fsync
import matplotlib.pyplot as plt

import functools
import operator
import optax
from jax import tree_util as tu
import chex

from jax.lax import stop_gradient as sg

def breakpoint_if_nonfinite(x):
  is_finite = jnp.isfinite(x).all()
  def true_fn(x):
    pass
  def false_fn(x): 
    jax.debug.breakpoint()
  jax.lax.cond(is_finite, true_fn, false_fn, x)


def conjugate_gradient_hvp(HVP, b, max_iter=10):
    """
    [Adapted using ChatGPT and Wiki: https://en.wikipedia.org/wiki/Conjugate_gradient_method]
    Conjugate Gradient solver for Hessian inverse-vector product.
    
    Args:
    - HVP: Function that computes the Hessian-vector product (HVP) as HVP(x) for a given vector x.
    - b: The right-hand side vector b as a JAX array.
    - x0: Initial guess for the solution as a JAX array.
    - tol: Tolerance for convergence (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 10).
    
    Returns:
    - x: The approximate solution to HVP(x) = b as a JAX array.
    """
    x = b  # Initial guess is b itself, (i.e., assume Hessian is identity)
    r = tree_add(b, HVP(x), scale_rhs=-1)  
    p = r
    rs = tree_dot(r, r)

    def cg_update(carry, _):
        x, r, p, rs = carry
        p = sg(p)
        Ap = HVP(p)
        
        alpha = rs / (tree_dot(p, Ap) + 1e-10)
        x = tree_add(x, p, scale_rhs=alpha)
        r = tree_add(r, Ap, scale_rhs=-alpha)
        rs_new = tree_dot(r, r)
        p = tree_add(r, p, scale_rhs=(rs_new / (rs + 1e-10)))
        return (x, r, p, rs_new), None

    (x, _, _, _) = jax.lax.scan(cg_update, (x, r, p, rs), xs=None, length=max_iter)[0]

    return x


def neumann_hvp(HVP, b, alpha=5e-1, max_iter=10, debug=False):
    """
    http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf  <- we use this one
    https://arxiv.org/pdf/1703.04730.pdf
    Both require 1 HVP and two additions and 2x parameter memory
    """
    p = b # Initial guess is b itself, (i.e., assume Hessian is identity)
    v = b

    def neumann_nuisance(carry, _):
        p, v = carry
        Hv = HVP(v)
        v_new = tree_add(v, Hv, scale_rhs=-alpha)     # v = (I-\alpha H)v
        v_new = sg(v_new) 

        if debug > 0:
            # Alpha should be chosen such that the max eigen value of H is near 1.
            # Choosing alpha too small, will require more iterations 
            # And choosing alpha not small enough will make iterations invalid
            # as max eigenvalue of H needs to be < 1 for the recursion to be valid
            # (Lowest eigen value of H >=0, as H is PSD.)
            # 
            # This is a neccessary condition, not a sufficient condition.
            # Therefore this is not a perfect check, but a good proxy.
            # 
            pass 

        p_new = tree_add(p, v_new)                     # p = p + v
        return (p_new, v_new), None
    
    (p, v) = jax.lax.scan(neumann_nuisance, (p, v), xs=None, length=max_iter)[0] 
    if max_iter > 0:
        p = tree_add(p, p, scale_lhs=alpha/2, scale_rhs=alpha/2)  # return alpha * p
    return p


class Checkpointer:
  def __init__(self, path='', writeout=False):
    if path == '':
        self.root = Path(os.getcwd())
        self.path = self.root / path
    else:
        self.path = path
    self.writeout = writeout
    self.cache = None

  def save(self, params):
    params = jax.device_get(params)

    if self.writeout:
      with open(self.path, 'wb') as fp:
        pickle.dump(params, fp)
    else:
      self.cache = copy.copy(params)

  def load(self):
    if self.writeout:
      with open(self.path, 'rb') as fp:
        params = pickle.load(fp)
    else:
      params = copy.copy(self.cache)

    return jax.device_put(params)
  
def average_plot(xs, ys, color, linewidth=2, alpha=0.2, label=None):
    mu, stderr = np.mean(ys, axis=0), 2*np.std(ys, axis=0)/np.sqrt(ys.shape[0])
    plt.plot(xs, mu, label=label, c=color, linewidth=linewidth)  
    plt.fill_between(xs, mu - stderr, mu + stderr, alpha=alpha, facecolor=color)

def normalize(matrix, ignore_last=False):
        
    n, d = matrix.shape
    k = d-1 if ignore_last else d

    mu = np.mean(matrix, axis=0, keepdims=True)
    std = np.std(matrix, axis=0, keepdims=True)

    matrix[:, :k] = (matrix[:, :k] - mu[:, :k]) / std[:, :k] #TODO
    if ignore_last:
        matrix[:, -1] = matrix[:, -1] # / np.sqrt(n) 

    assert matrix.shape == (n, d)
    return matrix


def entropy(probs):
    entropy = - jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1) 
    entropy = jnp.mean(entropy)
    entropy = entropy / jnp.log(probs.shape[-1])          # Normalize by max entropy value (i.e., log(n))
    return entropy


def binary_entropy(probs):
    entropy = - jnp.mean(probs * jnp.log(probs + 1e-4) + (1-probs) * jnp.log((1-probs) + 1e-4) ) 
    entropy = entropy / jnp.abs(jnp.log(probs.shape[-1] + 1e-4))          # Normalize by max entropy value (i.e., log(n))
    return entropy

def split_data(data):
   n = data[0].shape[0]

   data_1 = [item[:n//2] for item in data]
   data_2 = [item[n//2:] for item in data]

   return data_1, data_2


_vdot = functools.partial(jnp.vdot, precision=jax.lax.Precision.HIGHEST)

def _vdot_safe(a, b):
  return jnp.vdot(a, b)
  # return jnp.vdot(jnp.asarray(a), jnp.asarray(b))
#   return _vdot(jnp.asarray(a), jnp.asarray(b))


def tree_dot(tree_x, tree_y):
  """Compute the inner product <tree_x, tree_y>.
  https://jaxopt.github.io/stable/_modules/jaxopt/_src/tree_util.html"""
  vdots = tu.tree_map(_vdot_safe, tree_x, tree_y)
  return tu.tree_reduce(operator.add, vdots)

def tree_add(tree_x, tree_y, scale_rhs=1.0, scale_lhs=1.0):
    # https://github.com/deepmind/optax/blob/master/optax/_src/update.py#L24#L44
    return jax.tree_util.tree_map(
      lambda x, y: jnp.asarray(scale_lhs* x + scale_rhs* y).astype(jnp.asarray(y).dtype),
      tree_x, tree_y)
    # return jax.tree_util.tree_map(lambda x, y: x + y, tree_x, tree_y)

def clip_norm(updates, max_norm):
    """adapted from 
    https://github.com/deepmind/optax/blob/master/optax/_src/clipping.py#L91#L125
    """
    g_norm = optax._src.linear_algebra.global_norm(updates)
    g_norm = jnp.maximum(max_norm, g_norm)
    updates = jax.tree_util.tree_map(lambda t: (t / g_norm) * max_norm, updates)
    return updates



class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore, method):
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        if self.file:
            self.log.flush()
            fsync(self.log.fileno())



def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location


def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    dir_path = str.split(dir_path, sep='/')[1:]  #Ignore the blank characters in the start 
    for i in range(len(dir_path)):
        check_n_create(path.join('/', *(dir_path[:i + 1])))

def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
