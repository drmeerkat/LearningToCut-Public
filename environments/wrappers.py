import gym
import numpy as np

from gym.spaces import Tuple, Discrete, Box

class ObsPadding(gym.ObservationWrapper):
    # Just place holders to be compatible with gym vecenv
    metadata = {}
    
    def __init__(self, env, n_vars, n_cons, timelimit) -> None:
        super().__init__(env)
        self.n_vars = n_vars
        self.n_cons = n_cons
        self.epi_len = timelimit
        self.cons_shape = (self.n_cons + self.epi_len, self.n_vars)
        self.cuts_shape = (self.n_vars + self.n_cons + self.epi_len - 1, self.n_vars)
        self.observation_space = Tuple((
                Box(low=-np.inf, high=np.inf, shape=self.cons_shape), 
                Box(low=-np.inf, high=np.inf, shape=self.cuts_shape)
            )
        )
        self.action_space = Discrete(self.n_vars + self.n_cons + self.epi_len)

    def observation(self, obs):
        # Zero pad the input observation to be exactly the same shape
        A, b, c0, cuts_a, cuts_b = obs
        # by default, np.pad will zero-pad the array
        cons = np.concatenate((A, b.reshape(-1, 1)), axis=1)
        padded_cons = np.concatenate((cons, np.zeros(shape=(self.cons_shape[0] - cons.shape[0], self.n_vars))), axis=0)
        cuts = np.concatenate((cuts_a, cuts_b.reshape(-1, 1)), axis=1)
        padded_cuts = np.concatenate((cuts, np.zeros(shape=(self.cuts_shape[0] - cuts.shape[0], self.n_vars))), axis=0)
        cuts_mask = np.concatenate((np.ones(cuts.shape[0]), np.zeros(self.cuts_shape[0] - cuts.shape[0])))
        return (padded_cons, padded_cuts, cuts_mask)
