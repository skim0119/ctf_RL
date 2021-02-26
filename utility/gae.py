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
    discount_adv=True,
    normalize=False,
    td_lambda=False,
    standardize_td=False,
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
    if discount_adv:
        advantages = discount(advantages, gamma * lambd)

    if td_lambda:
        td_target = advantages + value_ext[:-1]
    if standardize_td:
        td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-9)

    # Normalize Advantage to be Normal
    if normalize:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-9)

    return list(td_target), list(advantages)
