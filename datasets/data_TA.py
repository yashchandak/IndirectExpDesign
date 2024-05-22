import numpy as np
import pandas as pd
import locale
import scipy
import matplotlib.pyplot as plt


class TripAdvisor():
    """
    This domain is adapted from EconML library
    https://github.com/py-why/EconML/blob/main/prototypes/dml_iv/TA_DGP_analysis.ipynb
    """

    def __init__(self, config=None) -> None:
        self.config = config

        self.n_IV = config.n_IV
        self.maxx_vals = np.array([28, 28, 28, 28, 28, 28, 1, 1, 2, 5e3]).reshape(1, -1)
        self.maxy_val, self.miny_val = 26, -12
        self.X_colnames = {
            'days_visited_exp_pre': 'day_count_pre',  # How many days did they visit TripAdvisor attractions pages in the pre-period
            'days_visited_free_pre': 'day_count_pre',  # How many days did they visit TripAdvisor through free channels (e.g. domain direct) in the pre-period
            'days_visited_fs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor fs pages in the pre-period    
            'days_visited_hs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor hotels pages in the pre-period
            'days_visited_rs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor restaurant pages in the pre-period
            'days_visited_vrs_pre': 'day_count_pre',  # How many days did they visit TripAdvisor vrs pages in the pre-period
            'is_existing_member': 'binary', #Binary indicator of whether they are existing member
            'locale_en_US': 'binary',  # User's locale
            'os_type': 'os',  # User's operating system
            'revenue_pre': 'revenue',  # Revenue in the pre-period
        }

        self.treat_colnames = {
            'treatment': 'binary',  # Did they receive the easier sign-up process in the experiment? [This is the instrument]
            'is_member': 'is_member'  # Did they become a member during the experiment period (through any means)? [This is the treatment of interest]
        }

        self.outcome_colnames = {
            'days_visited': 'days_visited',  # How many days did they visit TripAdvisor in the experimental period
        }

    def normalize(self, X, Y=None):
        Xnorm = X/self.maxx_vals
        assert np.shape(Xnorm) == np.shape(X)

        if Y is None:
            return Xnorm

        else:
            # Ynorm = Y 
            Ynorm = (Y - self.miny_val)/ (self.maxy_val - self.miny_val)
            return Xnorm, Ynorm

    def reset(self):
        pass

    def gen_data(self, data_type, n):
        gen_func = {'day_count_pre': lambda: np.random.randint(0, 29 , n),  # Pre-experiment period was 28 days
                    'day_count_post': lambda: np.random.randint(0, 15, n),  # Experiment ran for 14 days
                    'os': lambda: np.random.choice(['osx', 'windows', 'linux'], n),
                    'locale': lambda: np.random.choice(list(locale.locale_alias.keys()), n),
                    'count': lambda: np.random.lognormal(1, 1, n).astype('int'),
                    'binary': lambda: np.random.binomial(1, .5, size=(n,)),
                    ##'days_visited': lambda: 
                    'revenue': lambda: np.clip(np.round(np.random.lognormal(0, 3, n), 2), a_min=0, a_max=5e3)
                }
        
        return gen_func[data_type]() if data_type else None


    def get_covariates(self, n):
        X_data = {colname: self.gen_data(datatype, n) for colname, datatype in self.X_colnames.items()}
        X_data=pd.DataFrame({**X_data})
        # Turn strings into categories for numeric mapping
        X_data['os_type'] = X_data.os_type.astype('category').cat.codes
        return X_data.values.astype('float')


    def true_fn(self, X, T):
        temp =  .8 + .5*(5*(X[:,0]>5) + 10*(X[:,0]>15) + 5*(X[:, 0]>20)) - 3*X[:, 6]
        return temp * T + 5*(X[:, 0]>0) + 0.1*0.5


    def dgp_binary(self,X,Z):
        # Used for the plots with varying degree of confounder strength
        n = X.shape[0]
        
        nu = np.random.uniform(-5, 5, size=(n,)) # confounder
        
        coef_Z = 1 # 0.8

        # This is the original one for binary IV
        # C = np.random.binomial(1, coef_Z*scipy.special.expit(0.4*X[:, 0] + nu)) # Compliers when recomended
        # C0 = np.random.binomial(1, .006*np.ones(X.shape[0])) # Non-compliers when not recommended 
        # T = C * Z + C0 * (1 - Z)

        # THe following generalizes the above setup to work with multiple IVs
        # The underlying proces is similar, where one IV is strong, 
        # the rest are neutral. 
        temp = coef_Z*scipy.special.expit(0.4*X[:, 0] + nu)
        compliance = np.ones_like(Z) * .006 #TODO 
        compliance[:, 0] = temp + (1 - temp) * (1 - self.config.conf_strength)
        compliance[:, 1] = .006

        Tprobs = np.sum(Z * compliance, axis=-1)#, keepdims=True)  
        assert Tprobs.shape == (n,)
        T =  np.random.binomial(1, Tprobs).astype(np.float32) 
        assert T.shape == (n,)
        
        exp_y = self.true_fn(X, T)  
        assert exp_y.shape == (n,)
                
        # Noise heteroskedastic in treatments
        noise = np.random.uniform(-5, 5, size=(n,))
        y_noise =  T * noise * 0.01  + (1 - T) * noise * 2

        y =  exp_y + 2*nu + y_noise

        return y.reshape(n, 1), T.reshape(n, 1), exp_y.reshape(n, 1)


    def get_data(self, 
                 n, 
                 sampler=None, 
                 uniform=False,
                 test=False):
        
        X_pre = self.get_covariates(n)

        if uniform:
            probs = np.ones((n, self.n_IV)) / self.n_IV  
        elif self.config.algo_name=='oracle':
            probs = np.zeros((n, self.n_IV)) 
            probs[:, 0] = 0.5
            probs[:, 1] = 0.5
        else:
            probs = sampler(self.normalize(X_pre)) 
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
        assert (beta_p.shape == (n,1))


        Y_pre, T, exp_y = self.dgp_binary(X_pre, Z)

        # Do this after estimating Y, T, Z from X_pre
        X, Y = self.normalize(X_pre, Y_pre)

        train_data = (X, Z, T, Y, beta_p) 
        train_data = [item.astype('float32')  for item in train_data]
        # train_data = (X, Z, T, Y, beta_p, exp_y) 

        if test:
            Dx_n = 1000
            Dx_pre = self.get_covariates(n=Dx_n)
            Dt = np.random.binomial(1, p=0.5, size=Dx_n)
            Dy_pre = self.true_fn(Dx_pre, Dt) 
            Dx, Dy = self.normalize(Dx_pre, Dy_pre)
            test_data = ((Dx.astype('float32'), Dt.reshape(Dx_n, 1).astype('float32')), Dy.reshape(Dx_n, 1).astype('float32'))
            return train_data, test_data

        else:
             return train_data


class config:
    n_IV= 5

def test_TA():
    np.random.seed(4)
    dgp = TripAdvisor(config())
    X, Z, A, R , beta_p, exp_r = dgp.get_data(10000, uniform=True)

    print(X.shape, Z.shape, A.shape, R.shape, beta_p.shape, exp_r.shape)

    print("range X: ", np.max(X, axis=0), np.min(X, axis=0))
    print("range Y: ", np.max(R, axis=0), np.min(R, axis=0))

    plt.figure()
    idxs = np.where(A == 0)[0]
    xs = X[idxs, 0].reshape(-1, 1)
    plt.scatter(xs, exp_r[idxs], color='red') 
    plt.scatter(xs, R[idxs], color='red', alpha=0.2) 

    # Note that the true function depends on X[:,0] and X[:,6]
    # And we are aonly plotting for X[:,0], therefore for the same
    # X value on the plot we might see two/multiple values of "expected" Y
    idxs = np.where(A == 1)[0]
    xs = X[idxs, 0].reshape(-1, 1)
    plt.scatter(xs, exp_r[idxs], color='blue') 
    plt.scatter(xs, R[idxs], color='blue', alpha=0.2)

    plt.plot()

if __name__ == "__main__":
    test_TA()