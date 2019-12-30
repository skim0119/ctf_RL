import numpy as np

from utility.utils import discount_rewards

def gae(reward_list, value_list, bootstrap, gamma:float, lambd:float, normalize=True):
    """ gae

    Generalized Advantage Estimator

    Parameters
    ----------------
    reward_list: list
    value_list: list
    bootstrap: float
    gamma: float
    lambd: float
    normalize: boolean (True)

    Returns
    ----------------
    td_target: list
    advantage: list
    """
    reward_np = np.array(reward_list)
    value_ext = np.array(value_list+[bootstrap])

    td_target  = reward_np + gamma * value_ext[1:]
    advantages = reward_np + gamma * value_ext[1:] - value_ext[:-1]
    advantages = discount_rewards(advantages, gamma*lambd)

    if normalize:
        advantages = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-6)

    return td_target.tolist(), advantages.tolist()
