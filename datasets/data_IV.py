import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Toy_data():
    def __init__(self, config) -> None:
        self.config = config
    
        self.n_IV = config.n_IV
        # self.TRUE_PARAM=np.array([[1.0]])
        self.TRUE_PARAM=np.array([[1.0], [-1.0]])

        self.TRUE_COMPLIANCE = np.ones((self.n_IV, 1)) * 0.5

        useful_frac = 0.05
        n_useful_IV = max(int(useful_frac * self.n_IV), 1)   
        self.n_useful_IV = n_useful_IV

        # Note that this is not the actual "compliance"
        # actual "complaince" would correspond to E[XZ], that depends on conf_strength as well.
        self.TRUE_COMPLIANCE[:n_useful_IV] =   1.0  # Initial ones have the highest value
        self.TRUE_COMPLIANCE[-n_useful_IV:] =  0.0  # Last ones have the lowest value

    def reset(self):
        pass

    def get_true_params(self):
        return self.TRUE_PARAM

    def get_data(self, n, sampler=None, uniform=False, test=False):
       
        if uniform:
            probs = np.ones(self.n_IV) / self.n_IV
        elif self.config.algo_name == 'oracle':
            # This is oracle if the outcome noises are homoskedastic 
            probs = np.zeros(self.n_IV)
            
            # # Oracle for confounder strength 0.1
            probs[0] = 0.95
            probs[-1] = 0.05  
        else:   
            probs = sampler()
            
            # he output of my sorfmax() is in float32 type, however,
            # numpy.random.multinomial() will cast the pval into float64 type IMPLICITLY.
            # This data type casting would cause pval.sum() exceed 1.0 sometimes
            # due to numerical rounding.
            # https://stackoverflow.com/questions/23257587/how-can-i-avoid-value-errors-when-using-numpy-random-multinomial
            probs = np.asarray(probs).astype('float64')
            probs = probs / np.sum(probs)
    
        if self.config.debug == 1: 
            print("Probs: ", probs)

        confounder = np.random.randn(n, 1) * self.config.conf_strength #* 1 #* 0.01 #TODO

        Z = np.random.multinomial(1, probs.reshape(-1), size=n).astype(np.float32)
        assert np.shape(Z) == (n, self.n_IV)

        # Store the probabilities with which each Z was sampled
        beta_p = np.dot(Z, probs.reshape(-1,1))
        assert np.shape(beta_p) == (n, 1)
        
        Xprobs = np.clip(np.dot(Z, self.TRUE_COMPLIANCE) + confounder, a_min=0, a_max=1) 
        assert np.shape(Xprobs) == (n, 1)

        x = np.random.binomial(1, Xprobs).astype(np.float32)
        X = np.hstack([x, 1-x]) # Having X 2 dim necessitates playing atleast 2 instruments

        exp_y = np.dot(X, self.TRUE_PARAM)
        assert np.shape(exp_y) == (n, 1)
        
        y_noise = x * np.random.randn(n, 1) * 1 + (1 - x) * np.random.randn(n, 1) * 0.1
        assert np.shape(y_noise) == (n, 1)

        Y = exp_y + confounder + y_noise  # Heterskedastic noise

        train_data = (X, Y, Z, beta_p)

        if test:
            # For closedform
            return train_data, self.get_true_params() 
    
        else:
            return train_data


def test():
    # orig_x, orig_y, orig_exp_y, orig_z = get_data(N)

    # Normalize data, makes the hyper-parameters of the algorithm robust to scale
    # x, y, exp_y, z = normalize(orig_x), normalize(orig_y), normalize(orig_exp_y), orig_z
    # plt.scatter(x, y, s=5)
    # plt.plot(x, exp_y, c='red', linewidth=1)
    pass

if __name__ == "__main__":
    test()
