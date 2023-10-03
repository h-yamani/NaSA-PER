
import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import logging

from nasa_td3 import AE_TD3
from dm_control import suite
from utils.Frame_Stack import FrameStack
#from cares_reinforcement_learning.memory import MemoryBuffer
from utils.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from utils.PER import PER


def save_reward_values(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"data_results/{filename}", index=False)


def plot_reconstruction_img(original, reconstruction):
    input_img      = original[0]/255
    reconstruction = reconstruction[0]
    difference     = abs(input_img - reconstruction)

    plt.subplot(1, 3, 1)
    plt.title("Image Input")
    plt.imshow(input_img, vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.title("Image Reconstruction")
    plt.imshow(reconstruction, vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(difference, vmin=0, vmax=1)
    plt.pause(0.01)



def train(env, agent, file_name, intrinsic_on, number_stack_frames, args):

    # Training-parameters
    # ------------------------------------#
    max_steps_training    = args.max_steps_training
    max_steps_exploration = args.max_steps_pre_exploration  # max_steps_pre_exploration

    batch_size = args.batch_size
    G          = args. G
    k          = number_stack_frames
    # ------------------------------------#
    # RD=PER parameters
    alpha = 0.7  # 0.4 0.6
    min_priority = 1
    novelty_surprise = 0

    # Action size and format
    # ------------------------------------#
    action_spec      = env.action_spec()
    action_size      = action_spec.shape[0]    # For example, 6 for cheetah


    # Needed classes
    # ------------------------------------#
    #memory       = MemoryBuffer()
    frames_stack = FrameStack(env, k)
    # ------------------------------------#

    # Training Loop
    # ------------------------------------#
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    historical_reward_training   = {"step": [], "episode_reward": []}
    historical_reward_evaluation = {"step": [], "avg_episode_reward": []}

    # To store zero at the beginning
    historical_reward_evaluation["step"].append(0)
    historical_reward_evaluation["avg_episode_reward"].append(0)

    start_time = time.time()
    state      = frames_stack.reset()  # unit8 , (9, 84 , 84)
    #memory = PrioritizedReplayBuffer()
    memory = PER()
    #print(f'State: {args.latent_size}')

    for total_step_counter in range(1, int(max_steps_training) + 1):
        episode_timesteps += 1

        if total_step_counter <= max_steps_exploration:
            print(f"Running Pre-Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action = np.random.uniform(-1, +1, size=action_size)
        else:
            action = agent.select_action_from_policy(state)

        next_state, reward_extrinsic, done = frames_stack.step(action)
        a = 1.0
        b = 1.0
        surprise_rate, novelty_rate = agent.get_intrinsic_values(state, action, next_state)
        reward_surprise = surprise_rate * a
        reward_novelty = novelty_rate * b
        novelty_surprise = reward_novelty + reward_surprise

        if intrinsic_on and total_step_counter > max_steps_exploration:
            total_reward    = reward_extrinsic + reward_surprise + reward_novelty

        else:
            total_reward = reward_extrinsic
        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done, novelty_surprise=novelty_surprise)
        state = next_state
        episode_reward += reward_extrinsic  # just for plotting/comparing purposes use this reward as it is i.e. from env

        if total_step_counter > max_steps_exploration:
            for _ in range(G):
                experience = memory.sample(batch_size)
                # Unpack the experience tuple
                states, actions, rewards, next_states, dones, novelty_surprise, indices, weights = experience

                ''' # Initialize lists to collect surprise_rates and novelty_rates for each experience
                surprise_rates_batch = []
                novelty_rates_batch = []

                # Iterate over each experience in the batch
                for i in range(batch_size):
                    state = states[i]
                    action = actions[i]
                    next_state = next_states[i]
                    done = dones[i]
                    # Calculate surprise_rate and novelty_rate for the current experience
                    surprise_rate, novelty_rate = agent.get_intrinsic_values(state, action, next_state)

                    # Append the rates to the batch lists
                    surprise_rates_batch.append(surprise_rate)
                    novelty_rates_batch.append(novelty_rate)

                # Convert the lists to NumPy arrays for further processing if needed
                surprise_rates_batch = np.array(surprise_rates_batch)
                novelty_rates_batch = np.array(novelty_rates_batch)  '''
                # Calculate priority and update buffer
                # Combine surprise_rates_batch and novelty_rates_batch into a single NumPy array
                #combined_array = np.maximum(surprise_rates_batch + novelty_rates_batch, min_priority)
                combined_array = np.maximum(novelty_surprise, min_priority)

                # Create a PyTorch tensor from the combined NumPy array
                priority = torch.FloatTensor(combined_array).pow(alpha).cpu().data.numpy().flatten()

                #priority = (surprise_rates_batch+novelty_rates_batch).clamp(min=min_priority).pow(alpha).cpu().data.numpy().flatten()
                #print(f'priority:{priority.shape}finish')

                memory.update_priority(indices, priority)

                #states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size)

                agent.train_policy((
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ))

                if intrinsic_on:
                    agent.train_predictive_model((
                        states,
                        actions,
                        next_states
                    ))
            print(f'reward: {reward_extrinsic} total_reward:{total_reward}, Total step:{total_step_counter}')

        if done:
            episode_duration = time.time() - start_time
            print(f"TRAIN T:{total_step_counter} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Sec")

            historical_reward_training["step"].append(total_step_counter)
            historical_reward_training["episode_reward"].append(episode_reward)

            if total_step_counter % 10_000 == 0:
                print("*************--Evaluation Loop--*************")
                save_reward_values(historical_reward_training, filename=file_name + "_training")
                evaluation_loop(env, agent, frames_stack, total_step_counter, file_name, historical_reward_evaluation, args)
                print("--------------------------------------------")

            start_time = time.time()
            state = frames_stack.reset()
            episode_reward     = 0
            episode_timesteps  = 0
            episode_num       += 1

    agent.save_models(filename=file_name)
    save_reward_values(historical_reward_training, file_name + "_training")
    print("All GOOD AND DONE :)")



def evaluation_loop(env, agent, frames_stack, total_counter, file_name, historical_reward_evaluation, args):
    fps        = 30
    video_name = f'videos_evaluation/{file_name}_{total_counter}.mp4'
    frame = grab_frame(env)
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    historical_episode_reward = []
    number_eval_episodes      = args.number_eval_episodes

    for episode_num in range(number_eval_episodes):
        start_time = time.time()
        state      = frames_stack.reset()
        done       = False
        episode_reward    = 0
        episode_timesteps = 0

        while not done:
            if episode_num == 0:
                video.write(grab_frame(env))
            episode_timesteps += 1
            action = agent.select_action_from_policy(state, evaluation=True)
            state, reward_extrinsic, done = frames_stack.step(action)
            episode_reward += reward_extrinsic

        episode_duration = time.time() - start_time
        print( f" EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Sec")
        historical_episode_reward.append(episode_reward)

    mean_reward_evaluation = np.round(np.mean(historical_episode_reward), 2)
    historical_reward_evaluation["avg_episode_reward"].append(mean_reward_evaluation)
    historical_reward_evaluation["step"].append(total_counter)

    save_reward_values(historical_reward_evaluation, file_name +"_evaluation")
    video.release()



def grab_frame(env):
    frame = env.physics.render(camera_id=0, height=240, width=300)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame



def define_parse_args():
    parser = ArgumentParser()

    parser.add_argument('--max_steps_training',        type=int, default=1000000)
    parser.add_argument('--max_steps_pre_exploration', type=int, default=1000)
    parser.add_argument('--number_eval_episodes',      type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--G',          type=int, default=5)


    parser.add_argument('--intrinsic', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=200)
    parser.add_argument('--env',  type=str, default="ball_in_cup")
    parser.add_argument('--task', type=str, default="catch")
    args   = parser.parse_args()
    return args


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Working with = {device}")

    args = define_parse_args()
    domain_name = args.env
    task_name   = args.task
    seed        = args.seed
    print(f" Environment and Task Selected: {domain_name}_{task_name}")

    # ------------------------------------------------#
    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})
    action_spec = env.action_spec()
    action_size = action_spec.shape[0]
    latent_size = args.latent_size
    number_stack_frames = 3
    # ------------------------------------------------#

    # set seeds
    # ------------------------------------------------#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ------------------------------------------------#

    # ------------------------------------------------#
    # Create Directories
    print(f" Creating Folders")

    dir_exists = os.path.exists("videos_evaluation")
    if not dir_exists:
        os.makedirs("videos_evaluation")

    dir_exists = os.path.exists("data_results")
    if not dir_exists:
        os.makedirs("data_results")

    dir_exists = os.path.exists("models")
    if not dir_exists:
        os.makedirs("models")
    # ------------------------------------------------#

    # ------------------------------------------------#
    print(f" Initializing Algorithm.....")
    agent = AE_TD3(
        latent_size=latent_size,
        action_num=action_size,
        device=device,
        k=number_stack_frames)

    intrinsic_on  = args.intrinsic
    if intrinsic_on:
        print(f"Working with Autoencoder-TD3 and Novelty/Surprise Index")
    else:
        print(f"Working with Autoencoder-TD3")

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name = domain_name + "_" + task_name + "_" + "NASA_TD3" + "_" + "Intrinsic_" + str(intrinsic_on) + "_"  + str(date_time_str)
    print(f" File name for this training loop: {file_name}")

    print("Initializing Training Loop....")
    train(env, agent, file_name, intrinsic_on, number_stack_frames, args)


if __name__ == '__main__':
    main()
