import os

import time
import gym
import gym_cap
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

# the modules that you can use to generate the policy.
import policy.patrol
import policy.random
import policy.roomba
import policy.policy_A3C
import policy.zeros

import moviepy.editor as mp
from moviepy.video.fx.all import speedx

max_episode_length = 150

# Environment
env = gym.make("cap-v0").unwrapped  # initialize the environment
# policy_red = policy.policy_RL.PolicyGen(env.get_map, env.get_team_red,
#                                         model_dir='model_pretrain/A3C_CVT/',
#                                         input_name='global/actor/state:0',
#                                         output_name='global/actor/fully_connected/Softmax:0',
#                                         color='red'
#                                         )
policy_red = policy.zeros.PolicyGen(env.get_map, env.get_team_red)
policy_blue = policy.policy_A3C.PolicyGen(env.get_map, env.get_team_blue,
                                          model_dir='model/A3C_CTF_Zero/',
                                          input_name='global/state:0',
                                          output_name='global/actor/fully_connected_1/Softmax:0'
                                          )
observation = env.reset(map_size=20, policy_blue=policy_blue, policy_red=policy_red)

data_dir = 'clips'
total_run = 1000
num_success = 10
num_failure = 10
vid_success = []
vid_failure = []


def play_episode(frame_count, episode=0):
    # Set video recorder
    video_dir = os.path.join(data_dir, 'raw_videos')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_fn = 'episode_' + str(episode) + '.mp4'
    video_path = os.path.join(video_dir, video_fn)

    video_recorder = VideoRecorder(env, video_path)

    # Reset environmnet
    observation = env.reset()

    # Rollout episode
    episode_length = 0.
    done = 0
    while (done == 0):
        # set exploration rate for this frame
        video_recorder.capture_frame()
        episode_length += 1

        # state consists of the centered observations of each agent
        action = policy_blue.gen_action(env.get_team_blue, env._env)  # Full observability

        observation, reward, done, _ = env.step(action)

        # stop the episode if it goes too long
        if episode_length >= max_episode_length:
            reward = -100.
            done = True

    # Post Statistics
    success_flag = env.blue_win
    survival_rate = sum([agent.isAlive for agent in env.get_team_blue]) / len(env.get_team_blue)
    kill_rate = sum([not agent.isAlive for agent in env.get_team_red]) / len(env.get_team_red)

    # Closer
    video_recorder.close()
    vid = mp.VideoFileClip(video_path)

    if success_flag == 1 and len(vid_success) < num_success:
        vid_success.append(vid)
    elif success_flag == 0 and len(vid_failure) < num_failure:
        vid_failure.append(vid)

    return episode_length, reward, frame_count + episode_length, survival_rate, kill_rate, success_flag


def render_clip(frames, filename):
    vid = mp.concatenate_videoclips(frames)
    vid = speedx(vid, 0.1)

    final_vid = vid  # mp.clips_array([[legend, vid]])
    fp = os.path.join(data_dir, filename)
    final_vid.write_videofile(fp)


# Run
start_time = time.time()
episode = 0
done_flag = 0
frame_count = 0

length_list = []
reward_list = []
survive_list = []
kill_list = []
win_list = []

while ((len(vid_success) < num_success) or (len(vid_failure) < num_failure)) and episode < total_run:
    length, reward, frame_count, survival_rate, kill_rate, win = play_episode(frame_count, episode)

    # save episode data after the episode is done
    length_list.append(length)
    reward_list.append(reward)
    survive_list.append(survival_rate)
    kill_list.append(kill_rate)
    win_list.append(win)

    print('Mean Stat: Success: {} ---- Reward {} ---- Length: {}'
          .format(np.mean(win_list), np.mean(reward_list), np.mean(length_list)))
    print('            Survival Rate: {} ---- Kill Rate: {}'
          . format(np.mean(survive_list), np.mean(kill_list)))
    episode += 1

env.close()

render_clip(vid_success[0:num_success], 'success.mp4')
render_clip(vid_failure[0:num_failure], 'failure.mp4')
