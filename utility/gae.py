import numpy as np
import random
import scipy.signal


def discount(x, gamma):
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1], axis=-1)[::-1]


def gae(
    reward_list,
    value_list,
    bootstrap,
    gamma,
    lambd,
    terminal=False,
    mask=None,
    discount=True,
    normalize=True,
    td_lambda=False,
):
    """ gae

    Generalized Advantage Estimator

    Parameters
    ----------------
    reward_list: list
    value_list: list
    bootstrap: float
        Last critic value (0.0 if terminal)
    gamma: float
        Reward discount 
    lambd: float
        GAE discount 
    normalize: boolean (True)

    Returns
    ----------------
    td_target: list
    advantage: list
    """

    if terminal:
        bootstrap *= 0.0

    reward_np = np.array(reward_list)
    value_ext = np.array(value_list + [bootstrap])

    if mask is None:
        td_target = reward_np + gamma * value_ext[1:]
    else:
        td_target = reward_np + gamma * value_ext[1:] * (1 - np.array(mask))
    advantages = td_target - value_ext[:-1]

    # Discount Advantage (default: True)
    if discount:
        advantages = discount(advantages, gamma * lambd)

    if td_lambda:
        td_target = advantages + value_ext[:-1]

    # Normalize Advantage to be Normal
    if normalize:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

    return td_target.tolist(), advantages.tolist()
