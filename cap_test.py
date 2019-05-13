import time
import gym
import gym_cap
import numpy as np

# the modules that you can use to generate the policy.
import policy.patrol
import policy.random
import policy.roomba
import policy.policy_A3C
import policy.zeros


start_time = time.time()
env = gym.make("cap-v0").unwrapped  # initialize the environment

# reset the environment and select the policies for each of the team
policy_red = policy.zeros.PolicyGen(env.get_map, env.get_team_red)
policy_blue = policy.roomba.PolicyGen(env.get_map, env.get_team_blue)
# policy_red = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red,
#                                         model_dir='model_pretrain/A3C_CVT/',
#                                         input_name='global/actor/state:0',
#                                         output_name='global/actor/fully_connected/Softmax:0',
#                                         color='red'
#                                         )
#policy_blue = policy.policy_A3C.PolicyGen(env.get_map, env.get_team_blue,
#                                          model_dir='model/A3C_CTF_Roomba/',
#                                          input_name='global/state:0',
#                                          output_name='global/actor/fully_connected_1/Softmax:0'
#                                          )

observation = env.reset(map_size=20,
                        policy_blue=policy_blue,
                        policy_red=policy_red)


pre_score = 0
done = False
t = 0
total_score = 0
while True:
    t = 0
    while not done:
        action = policy_blue.gen_action(env.get_team_blue, env._env)  # Full observability
        observation, reward, done, info = env.step(action)

        # render and sleep are not needed for score analysis
        t += 1
        env.render(mode="fast")
        if t == 150:
            break

    env.reset()
    done = False
    print("Time: %.2f s, score: %.2f" %
          ((time.time() - start_time), reward))
