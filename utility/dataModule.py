import numpy as np
import collections
import gym_cap.envs.const as CONST

UNKNOWN = CONST.UNKNOWN            # -1
TEAM1_BG = CONST.TEAM1_BACKGROUND  # 0
TEAM2_BG = CONST.TEAM2_BACKGROUND  # 1
TEAM1_GV = CONST.TEAM1_UGV         # 2
TEAM1_UAV = CONST.TEAM1_UAV        # 3
TEAM2_GV = CONST.TEAM2_UGV         # 4
TEAM2_UAV = CONST.TEAM2_UAV        # 5
TEAM1_FL = CONST.TEAM1_FLAG        # 6
TEAM2_FL = CONST.TEAM2_FLAG        # 7
OBSTACLE = CONST.OBSTACLE          # 8
DEAD = CONST.DEAD                  # 9
SELECTED = CONST.SELECTED          # 10
COMPLETED = CONST.COMPLETED        # 11

SIX_MAP_CHANNEL = {UNKNOWN: 0, DEAD: 0,
                   TEAM1_BG: 1, TEAM2_BG: 1,
                   TEAM1_GV: 2, TEAM2_GV: 2,
                   TEAM1_UAV: 3, TEAM2_UAV: 3,
                   TEAM1_FL: 4, TEAM2_FL: 4,
                   OBSTACLE: 5}
SEVEN_MAP_CHANNEL = {UNKNOWN: 0, DEAD: 0,
                     TEAM1_BG: 1, TEAM2_BG: 1,
                     TEAM1_GV: 2,
                     TEAM2_GV: 3,
                     TEAM1_UAV: 4, TEAM2_UAV: 4,
                     TEAM1_FL: 5, TEAM2_FL: 5,
                     OBSTACLE: 6}

class fake_agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_loc(self):
        return (self.x, self.y)

def meta_state_processor(full_state, game_info=None, map_size=20, flatten=False, reverse=False):
    """meta_state_processor """
    if game_info is None:
        agent_locs = _find_coord(full_state, TEAM2_GV if reverse else TEAM1_GV)
        enemy_locs = _find_coord(full_state, TEAM1_GV if reverse else TEAM2_GV)
        agent_alive = [True]*len(agent_locs)
        enemy_alive = [True]*len(enemy_locs)
    else:
        agent_locs = game_info['blue_locs']
        enemy_locs = game_info['red_locs']
        agent_alive = game_info['blue_alive'][-1]
        enemy_alive = game_info['red_alive'][-1]

    # Build Shared State
    flag_loc = _find_coord(full_state, TEAM1_FL if reverse else TEAM2_FL, single_value=True)
    num_agent = len(agent_alive)
    num_alive_agent = sum(agent_alive)
    num_alive_enemy = sum(enemy_alive)
    shared_state = np.array([[*flag_loc, num_alive_agent, num_alive_enemy]] * num_agent)

    # Build Individual State
    oh_state = np.zeros((num_agent, *full_state.shape, 8))
    decomp_state = _decompose_full_state(full_state, reverse=reverse)
    for agent_state in oh_state:
        agent_state[:,:,:-1] = decomp_state
    for idx, (loc, alive) in enumerate(zip(agent_locs, agent_alive)):
        if not alive:
            continue
        oh_state[idx,loc[0],loc[1],-1] = 1

    if flatten:
        return np.reshape(oh_state, (num_agent, -1)), shared_state
    return oh_state, shared_state

def state_processor(state, agents=None, vision_radius=19, full_state=None, flatten=False, reverse=False, partial=True):
    """ pre_processor

    Return encoded state, position, and goal position
    """
    if not partial:
        state = full_state
    # Find Flag Location
    flag_id = TEAM1_FL if reverse else TEAM2_FL
    flag_locs = list(zip(*np.where(full_state == flag_id)))  
    if len(flag_locs) == 0:
        flag_loc = (-1,-1)
    else:
        flag_loc = flag_locs[0]

    # One-hot encode state
    oh_state = one_hot_encoder(state, agents, vision_radius, reverse, flatten=flatten)

    # gps state
    agents_loc = [agent.get_loc() for agent in agents]

    # Count number of enemy and allies
    items = collections.Counter(full_state.flatten())
    num_team1 = items[TEAM1_GV]
    num_team2 = items[TEAM2_GV]

    # Concatenate global status
    shared_status = np.concatenate([list(flag_loc), [num_team1], [num_team2]])

    indv_status = []
    for loc in agents_loc:
        status = np.concatenate([list(loc), [num_team1], [num_team2]])
        indv_status.append(status)

    return oh_state, indv_status, shared_status


def one_hot_encoder(state, agents=None, vision_radius=9,
                    flatten=False, locs=None):
    """Encoding pipeline for CtF state to one-hot representation

    6-channel one-hot representation of state.
    State is not binary: team2 is represented with -1.
    Channels are not symmetrical.

    :param state: CtF state in raw format
    :param agents: Agent list of CtF environment
    :param vision_radius: Size of the vision range (default=9)`
    :param reverse: Reverse the color. Used for red-perspective (default=False)
    :param flatten: Return flattened representation (for array output)
    :param locs: Provide locations instead of agents. (agents must be None)

    :return oh_state: One-hot encoded state
    """

    if agents is None:
        assert locs is not None
        agents = [fake_agent(x, y) for x, y in locs]

    vision_lx = 2 * vision_radius + 1
    vision_ly = 2 * vision_radius + 1
    oh_state = np.zeros((len(agents), vision_lx, vision_ly, 6), np.float64)

    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = SIX_MAP_CHANNEL
    map_color = {UNKNOWN: 1, DEAD: 0,
                 TEAM1_BG: 0, TEAM2_BG: 1,
                 TEAM1_GV: 1, TEAM2_GV: -1,
                 TEAM1_UAV: 1, TEAM2_UAV: -1,
                 TEAM1_FL: 1, TEAM2_FL: -1,
                 OBSTACLE: 1}

    # Expand the observation with wall to avoid dealing with the boundary
    sx, sy = state.shape
    _state = np.full((sx + 2 * vision_radius, sy + 2 * vision_radius), OBSTACLE)
    _state[vision_radius:vision_radius + sx, vision_radius:vision_radius + sy] = state
    state = _state

    for idx, agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x - vision_radius:x + vision_radius + 1, y - vision_radius:y + vision_radius + 1]  # extract view

        # FULL MATRIX OPERATION
        for channel, val in map_color.items():
            if val == 1:
                oh_state[idx, :, :, map_channel[channel]] += (vision == channel).astype(np.int32)
            elif val == -1:
                oh_state[idx, :, :, map_channel[channel]] -= (vision == channel).astype(np.int32)

    if flatten:
        return np.reshape(oh_state, (len(agents), -1))
    else:
        return oh_state

def _find_coord(grid, element_id, single_value=False):
    """_find_coord

    Given the 2d grid, return possible coordinates
    If none, return (-1,-1)

    :param grid:
        2D Grid
    :param element_id:
        Element number to find
    """
    # Find Location
    locs = list(zip(*np.where(grid== element_id)))
    if len(locs) == 0:
        return (-1,-1)
    if single_value:
        return locs[0]
    else:
        return locs

def _decompose_full_state(full_state, reverse=False):
    """_decompose_full_state"""

    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = SEVEN_MAP_CHANNEL
    channel_size=7
    map_color = {UNKNOWN: 1, DEAD: 0,
                 TEAM1_BG: 0, TEAM2_BG: 1,
                 TEAM1_GV: 1,
                 TEAM2_GV: 1,
                 TEAM1_UAV: 1, TEAM2_UAV: -1,
                 TEAM1_FL: 1, TEAM2_FL: -1,
                 OBSTACLE: 1}
    if reverse:
        map_color.update({TEAM1_BG: 1, TEAM2_BG: 0,
                          TEAM1_GV: -1, TEAM2_GV: 1,
                          TEAM1_UAV: -1, TEAM2_UAV: 1,
                          TEAM1_FL: -1, TEAM2_FL: 1})

    # Full matrix operation
    oh_state = np.zeros((*(full_state.shape) , channel_size), dtype=np.float)
    for channel, val in map_color.items():
        if val == 1:
            oh_state[:, :, map_channel[channel]] += (full_state == channel).astype(np.int32)
        elif val == -1:
            oh_state[:, :, map_channel[channel]] -= (full_state == channel).astype(np.int32)

    return oh_state

'''def no_com_encoder(state, agents=None, vision_radius=9, num_uav=0
                    flatten=False, locs=None):
    self.observation_space_blue = np.full_like(self._env, UNKNOWN)
    for agent in agents[-num_uav:]:
        # agent dies, loose vision
        if not agent.isAlive:
            continue
        loc = agent.get_loc()
        for i in range(-agent.range, agent.range + 1):
            for j in range(-agent.range, agent.range + 1):
                locx, locy = i + loc[0], j + loc[1]
                if (i * i + j * j <= agent.range ** 2) and \
                        not (locx < 0 or locx > self.map_size[0] - 1) and \
                        not (locy < 0 or locy > self.map_size[1] - 1):
                    self.observation_space_blue[locx][locy] = self._env[locx][locy]

    vision_lx = 2 * vision_radius + 1
    vision_ly = 2 * vision_radius + 1
    oh_state = np.zeros((len(agents), vision_lx, vision_ly, 6), np.float64)

    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = SIX_MAP_CHANNEL
    map_color = {UNKNOWN: 1, DEAD: 0,
                 TEAM1_BG: 0, TEAM2_BG: 1,
                 TEAM1_GV: 1, TEAM2_GV: -1,
                 TEAM1_UAV: 1, TEAM2_UAV: -1,
                 TEAM1_FL: 1, TEAM2_FL: -1,
                 OBSTACLE: 1}

    # Expand the observation with wall to avoid dealing with the boundary
    sx, sy = state.shape
    _state = np.full((sx + 2 * vision_radius, sy + 2 * vision_radius), OBSTACLE)
    _state[vision_radius:vision_radius + sx, vision_radius:vision_radius + sy] = state
    state = _state

    for idx, agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x - vision_radius:x + vision_radius + 1, y - vision_radius:y + vision_radius + 1]  # extract view

        # FULL MATRIX OPERATION
        for channel, val in map_color.items():
            if val == 1:
                oh_state[idx, :, :, map_channel[channel]] += (vision == channel).astype(np.int32)
            elif val == -1:
                oh_state[idx, :, :, map_channel[channel]] -= (vision == channel).astype(np.int32)

    if flatten:
        return np.reshape(oh_state, (len(agents), -1))
    else:
        return oh_state
        '''

def debug():
    """debug
    Include testing code for above methods and classes.
    The execution will start witn __main__, and call this method.
    """

    import gym
    import time
    env = gym.make("cap-v0")
    s = env.reset(map_size=20)

    print('start running')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')

    s = env.reset(map_size=20)

    print('start running v2')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder_v2(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')


if __name__ == '__main__':
    debug()
