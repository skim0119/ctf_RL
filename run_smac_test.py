from smac.env import StarCraft2Env
import numpy as np

def main():
    env = StarCraft2Env(map_name="2c_vs_64zg")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 1000


    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        mean_obs = []
        mean_state = []
        while not terminated:
            obs = np.array(env.get_obs())
            state = np.array(env.get_state())
            mean_obs.append(obs.mean())
            mean_state.append(state.mean())

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))
        #print("Battle won {}".format(env.battles_won))
        #print("Observation mean : {}, state mean : {}".format(np.mean(mean_obs), np.mean(mean_state)))

    env.close()

if __name__=="__main__":
    main()

    for i in range(len(env.agents)):
        agent = env.agents[i]
        agent.unit_type
