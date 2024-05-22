import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class Toy_data2():
    def __init__(self, config) -> None:
        self.config = config
    
        # self.TRUE_PARAM=np.array([[4.0], [0.0]])
        self.TRUE_PARAM=np.array([[4.0]])
        self.TRUE_COMPLIANCE = np.array([[0.1],[2.5],[10]])

    def reset(self):
        pass

    def get_true_params(self):
        return self.TRUE_PARAM

    def get_data(self, n, sampler=None, uniform=False, test=False):
       
        if uniform:
            # probs = np.array([0.5, 0., 0.5])
            probs = np.array([1/3.]*3)
        else:   
            probs = sampler()
    
        if self.config.debug == 1: 
            print("Probs: ", probs)
        
        confounder = np.random.randn(n, 1) * 10
        x_noise = np.random.randn(n, 1) * 10
        y_noise = np.random.randn(n, 1) * 10

        z = np.random.multinomial(1, probs.reshape(-1), size=n).astype(np.float32)
        x = np.dot(z, self.TRUE_COMPLIANCE) + confounder + x_noise
        # x = np.hstack([x, np.ones((len(x), 1))])
        exp_y = np.dot(x, self.TRUE_PARAM)#+ 5
        y = exp_y + confounder + y_noise  # Heterskedastic noise

        assert ((np.shape(x)[0],1) == np.shape(y)) 
        assert (np.shape(y) == np.shape(exp_y))

        # Store the probabilities with which each Z was sampled
        beta_p = np.dot(z, probs.reshape(-1,1))
        assert np.shape(beta_p) == (n, 1)

        # TODO: test data should contain the true param

        train_data = (x, y, z, beta_p)

        if test:
            # For closedform
            return train_data, self.get_true_params() #TODO
        
            # Otherwise
            # Dx = np.random.randn(100, 1) * 10
            Dx = np.array([[1]])
            Dy = Dx * self.TRUE_PARAM

            return train_data, (Dx, Dy)
    
        else:
            return train_data


class Toy_data():
    def __init__(self, config) -> None:
        self.config = config
    
        self.n_IV = config.n_IV

        # self.TRUE_PARAM=np.array([[4.0], [0.0]])
        self.TRUE_PARAM=np.array([[-4.0], [4.0]])

        self.TRUE_COMPLIANCE = np.ones((self.n_IV, 1)) * 0.5

        useful_frac = 0.05
        n_useful_IV = max(int(useful_frac * self.n_IV), 1)   
        self.n_useful_IV = n_useful_IV

        self.TRUE_COMPLIANCE[:n_useful_IV] = 0.9    # Initial ones have the highest value
        self.TRUE_COMPLIANCE[-n_useful_IV:] = 0.1  # Last ones have the lowest value
        # self.TRUE_COMPLIANCE[n_useful_IV:2*n_useful_IV] = 0.1  # Last ones have the lowest value
        # self.TRUE_COMPLIANCE[0] = 0.9   # Initial ones have the highest value
        # self.TRUE_COMPLIANCE[1] = 0.1  # Last ones have the lowest value

        
        # self.TRUE_COMPLIANCE =  np.array([[1.0],[0.1],[0.5]]) 
        # self.TRUE_COMPLIANCE =  np.array([[1.0],[0.1],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]]) 

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

            # Oracle for confounder strength 1
            # probs[0] = 0.45
            # probs[-1] = 0.55 
            probs[0] = 0.5 #TODO
            probs[-1] = 0.5 
            
            # # Oracle for confounder strength 0.1
            # probs[0] = 0.1
            # probs[-1] = 0.9 #TODO 

            # probs[1] = 0.55 #TODO
            # probs[self.n_useful_IV] = 0.55 
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
        # Store the probabilities with which each Z was sampled
        beta_p = np.dot(Z, probs.reshape(-1,1))
        assert np.shape(beta_p) == (n, 1)
        
        Xprobs = np.clip(np.dot(Z, self.TRUE_COMPLIANCE) + confounder, a_min=0, a_max=1) 
        assert np.shape(Xprobs) == (n, 1)

        x = np.random.binomial(1, Xprobs).astype(np.float32)
        X = np.hstack([x, 1-x]) 
        assert np.shape(X) == (n, 2)

        exp_y = np.dot(X, self.TRUE_PARAM)
        assert np.shape(exp_y) == (n, 1)
        
        y_noise = x * np.random.randn(n, 1) * 0.1 + (1 - x) * np.random.randn(n, 1) * 1 
        assert np.shape(y_noise) == (n, 1)

        Y = exp_y + confounder + y_noise  # Heterskedastic noise

        train_data = (X, Y, Z, beta_p)

        if test:
            # For closedform
            return train_data, self.get_true_params() #TODO
        
            # # Otherwise
            # # Dx = np.random.randn(100, 1) * 10
            # Dx = np.array([[1.0, 0.], [0., 1.0]])
            # Dy = np.dot(Dx, self.TRUE_PARAM)

            # return train_data, (Dx, Dy)
    
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
