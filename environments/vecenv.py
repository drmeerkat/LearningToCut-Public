import numpy as np
from copy import deepcopy

from gym.vector import SyncVectorEnv

__all__ = ["CustomSyncVecEnv"]


class CustomSyncVecEnv(SyncVectorEnv):
    """Vectorized environment that serially runs multiple environments.
    This is specifically modified for Learning2Cut environments!

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space, action_space, copy)

    def reset_wait(self):
        self._dones[:] = False
        cons, cuts, masks = [], [], []
        # No observation concatanation, just return a list!
        for env in self.envs:
            cur_cons, cur_cuts, cur_masks = env.reset()
            cons.append(cur_cons)
            cuts.append(cur_cuts)
            masks.append(cur_masks)
        self.observations = (np.stack(cons), np.stack(cuts), np.stack(masks))

        return deepcopy(self.observations) if self.copy else self.observations

    def step_wait(self):
        cons, cuts, masks = [], [], []
        infos = []
        # No observation concatanation, just return a list!
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if self._dones[i]:
                observation = env.reset()
            cur_cons, cur_cuts, cur_masks = observation
            cons.append(cur_cons)
            cuts.append(cur_cuts)
            masks.append(cur_masks)
            infos.append(info)
        self.observations = (np.stack(cons), np.stack(cuts), np.stack(masks))

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )
