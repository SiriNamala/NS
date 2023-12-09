import os
import time

import torch
import numpy as np
import pickle

#from actorCritic import ActorCritic
from main_fed_test_ppo import Scenario, ENV_ppo
from actorCritic import Actor, ActorCritic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

discountFactor = 0.99
steps = 20

sf = Scenario()
nagents = 1
length = 20

env = ENV_ppo(length, sf, nagents)




#train function to be called from main


def train_ac(Actor, batch_size=0, bplus=0, random_seed=0, initial_batch_list=[], last_evaluation_episodes=0, load=False, save=False, um=1):

    sf = Scenario()
    nagents = 1

    length = 20
    has_continuous_action_space = False

    env = ENV_ppo(length, sf, nagents)

    # state space dimension
    state_dim = env.length

    # action space dimension
    if has_continuous_action_space:
        action_dim = len(env.action_space)
    else:
        action_dim = len(env.action_space)

    # action space dimension
    action_dim = len(env.action_space)

    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = sum(initial_batch_list) + int(
        last_evaluation_episodes * state_dim - 1)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = max_training_timesteps  # save model frequency (in num timesteps)

    action_std = None

    batch_list = initial_batch_list + [int(last_evaluation_episodes * state_dim)]

    batch_idx = 0
    current_batch = batch_list[batch_idx]

    eval_set1, eval_set2 = set(), set()

    K_epochs = 10  # update policy for K epochs
    eps_clip = 0.5  # clip parameter for PPO
    gamma = 0.9  # discount factor

    lr_actor = 0.00005  # learning rate for actor network   /0.0008 0.0001
    lr_critic = 0.0005  # learning rate for critic network / 0.008 0.001

    random_seed = random_seed  # set random seed if required (0 = no random seed)


    #modify directory here ***
    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = [directory + "PPO_{}_{}_{}_{}.pth".format(env_name, random_seed, batch_size, i) for i in
                       range(nagents)]

    if random_seed:
        # print("--------------------------------------------------------------------------------------------")
        # print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # env.seed(random_seed)
        np.random.seed(random_seed)



    # initialize an Actor Critic agent agent
    #agents = [ActorCritic(nagents, state_dim, action_dim, lr_actor, lr_critic, gamma,
    #                  has_continuous_action_space, action_std) for i in range(nagents)]

    agents = [ActorCritic(nagents, state_dim, action_dim, lr_actor, lr_critic, gamma, has_continuous_action_space, action_std) for i in range(nagents)]


    if load:
        for i in range(1):
            agents[i].load(checkpoint_path[i])
    # track total training time
    start_time = time.time()

    # printing variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    update_step = 0
    i_episode = 0

    reward_list = []
    eval_list = []
    er_list = []
    re_list = []

    total_metric_list = []

    #def train_actor_critic(self, actor, critic, no_episodes, env):
    while time_step <= 9500:  # max_training_timesteps

            state = env.reset()
            total_reward = 0
            current_ep_reward = 0
            log_probs = []
            values = []
            rewards = []
            masks = []
            metric_list = []
            entropy = 0
            env.reset()

            for j in range(0, max_ep_len):
                actions = [agents[i].select_action(state, i) for i in range(nagents)]
                print("actions:", actions)

                state, reward, done = env.step(actions, eval_set1=eval_set1, eval_set2=eval_set2)

                for i in range(nagents):
                    state = torch.FloatTensor(state).to(device)
                    dist = Actor.actor(state)
                    value = Actor.critic(state)

                    action = dist.sample()
                    next_state, reward, done, _ = env.step(action.cpu().numpy())

                    log_prob = dist.log_prob(action).unsqueeze(0)
                    entropy += dist.entropy().mean()

                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                    masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

                    state = next_state

                    time_step += 1
                    update_step += 1
                    current_ep_reward += sum(reward)

                    next_state = torch.FloatTensor(next_state).to(device)
                    next_value = Actor.critic(next_state)
                    returns = Actor.return_rewards(next_value, rewards, masks)
                    log_probs = torch.cat(log_probs)
                    returns = torch.cat(returns).detach()
                    values = torch.cat(values)

                    advantage = returns - values

                    actor_loss = -(log_probs * advantage.detach()).mean()
                    critic_loss = advantage.pow(2).mean()

                    Actor.actor_optimizer.zero_grad()
                    Actor.critic_optimizer.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    Actor.actor_optimizer.step()
                    Actor.critic_optimizer.step()


if __name__ == "__main__":

    #train_actor_critic(actor, critic, length, env)
    rewards_list = []
    evals_list = []
    # metrics_list = []
    er_list = []
    re_list = []
    for i in range(10):
        reward_list, eval_list, metric_list, er, re = train_ac(initial_batch_list=initial_batch_list, random_seed=i,
                                                            um=arglist.um)
        rewards_list.append(reward_list)
        evals_list.append(eval_list)
        er_list.append(er)
        re_list.append(re)
        # metrics_list.append(metric_list)
        pickle.dump([rewards_list, er_list, re_list], open("reward_ppo.pkl", "wb"))














