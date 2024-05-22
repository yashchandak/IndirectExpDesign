import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Toy_data():
    def __init__(self, config=None) -> None:
        self.config = config
        
        self.n_IV = config.n_IV
    
        N = config.samples_first_batch + config.samples_per_batch * (config.n_batches - 1)
        self.TRUE_PARAM=np.array([[4.0]]) / np.sqrt(N)
        
        self.TRUE_COMPLIANCE = np.ones((self.n_IV, 1)) * 0.5
        useful_frac = 0.05
        n_useful_IV = max(int(useful_frac * self.n_IV), 1)   
        self.n_useful_IV = n_useful_IV

        self.TRUE_COMPLIANCE[:n_useful_IV] = 0.9    # Initial ones have the highest value
        self.TRUE_COMPLIANCE[-n_useful_IV:] = 0.1  # Last ones have the lowest value


    def reset(self):
        pass
    
    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))

    def get_true_params(self):
        return self.TRUE_PARAM

    def get_data(self, 
                 n, 
                 sampler=None, 
                 uniform=False,
                 test=False):
        
        def fn(n): 
            x0 = np.random.binomial(1, p=0.5, size=(n,1))    # Binary
            x1 = (np.random.uniform(size=(n,1))- 0.5) * 2    # Continuous [-1, 1] 
            X  = np.hstack([x0, x1])
            assert np.shape(X) == (n,2)
            return X

        X = fn(n)

        if uniform:
            probs = np.ones((n, self.n_IV)) / self.n_IV
        elif self.config.algo_name == 'oracle':
            raise ValueError("Oracle unkown")   
        else:
            probs = sampler(X) 
            assert np.shape(probs) == (n, self.n_IV)
            
            # he output of my sorfmax() is in float32 type, however,
            # numpy.random.multinomial() will cast the pval into float64 type IMPLICITLY.
            # This data type casting would cause pval.sum() exceed 1.0 sometimes
            # due to numerical rounding.
            # https://stackoverflow.com/questions/23257587/how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
            probs = np.asarray(probs).astype('float64')
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

        Z = np.array([np.random.multinomial(1, p).astype(np.float32) for p in probs])
        assert np.shape(Z) == np.shape(probs)

        # These are the probabilities for the samples corresponding to 1 or 0
        beta_p = np.sum(Z * probs, axis=-1, keepdims=True)
        assert np.shape(beta_p) == (n, 1)
 
        confounder = np.random.randn(n, 1) * self.config.conf_strength

        x1, x2 = X[:,:1], X[:, 1:]
        assert np.shape(x1) == (n,1)
        compliance = x1 @ self.TRUE_COMPLIANCE.T + (1 - x1) @ (1 - self.TRUE_COMPLIANCE).T    # Flip compliance based on binary covariate of x
        assert np.shape(compliance) == (n, self.n_IV)
        Aprobs = np.clip(np.sum(Z * compliance, axis=-1, keepdims=True) + confounder, a_min=0, a_max=1) 
        assert np.shape(Aprobs) == (n, 1)

        A = np.random.binomial(1, Aprobs)
        assert np.shape(A) == (n, 1)

        exp_y = A * self.TRUE_PARAM + x2 # Expected values varies with cont feature of x
        assert np.shape(exp_y) == (n, 1)
        
        y_noise = A * np.random.randn(n, 1) * 0.1 + (1 - A) * np.random.randn(n, 1) * 1.0 
        assert np.shape(y_noise) == (n, 1)

        R = exp_y + confounder + y_noise  # Treatment Heterskedastic noise

        train_data = (X, Z, A, R, beta_p) #, exp_r)
        train_data = [item.astype('float32')  for item in train_data]

        if test:            
            Dx_n = 100
            Dx = fn(Dx_n)
            Dt = np.random.binomial(1, p=0.5, size=(Dx_n, 1))
            Dy = Dt * self.TRUE_PARAM + Dx[:, 1:]

            return train_data, ((Dx.astype('float32'), Dt.astype('float32')), Dy.astype('float32') )
    
        else:
            return train_data
 


def test_CIV():
    dgp = Toy_data()
    X, Z, A, R , beta_p, exp_r = dgp.get_data(1000, uniform=True)

    print(X.shape, Z.shape, A.shape, R.shape, beta_p.shape, exp_r.shape)

    plt.figure()
    idxs = np.where(A == 0)[0]
    plt.scatter(X[idxs], exp_r[idxs], color='red') 

    plt.scatter(X[idxs], R[idxs], color='red', alpha=0.2) 

    idxs = np.where(A == 1)[0]
    plt.scatter(X[idxs], exp_r[idxs], color='blue') 
    plt.scatter(X[idxs], R[idxs], color='blue', alpha=0.2)

    plt.plot()

if __name__ == "__main__":
    test_CIV()