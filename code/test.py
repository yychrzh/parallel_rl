from model import *
import argparse

SAVE_PATH = ''
parser = argparse.ArgumentParser(description='TRPO.')

# these parameters should stay the same
parser.add_argument("--task", type=str, default='L2R')
parser.add_argument("--weights_path", type=str, default=SAVE_PATH + 'weights/')
parser.add_argument("--log_path", type=str, default=SAVE_PATH + 'logs/')
parser.add_argument("--memory_path", type=str, default=SAVE_PATH + 'memory/')
parser.add_argument("--timesteps_per_batch", type=int, default=2000)                   # default: 10000
parser.add_argument("--n_steps", type=int, default=200000)                             # default: 600000000
parser.add_argument("--gamma", type=float, default=.96)                                # default: 0.99
parser.add_argument("--max_kl", type=float, default=.01)                               # default: 0.001
parser.add_argument("--cg_damping", type=float, default=1e-3)                          # default: 1e-3
parser.add_argument("--num_threads", type=int, default=4)                              # default: 36
parser.add_argument("--monitor", type=bool, default=False)                             # default: False
parser.add_argument("--max_pathlength", type=int, default=200)

# change these parameters for testing
parser.add_argument("--decay_method", type=str, default="adaptive")  # adaptive, none
parser.add_argument("--timestep_adapt", type=int, default=0)  # 0
parser.add_argument("--kl_adapt", type=float, default=0)  # 0

args = parser.parse_args()


learner_tasks = multiprocessing.JoinableQueue()
learner_results = multiprocessing.Queue()
learner_env = WrapperEnv(visualize=False)

learner = PPO(args, learner_env.observation_space_shape, learner_env.action_space, learner_tasks, learner_results)
learner.make_model()
learner.loadModel("save_data/2018-04-11/data-36/weights/", 0)


def main():
    env = learner_env
    agent = learner
    observation = env.reset(difficulty=0)
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        action = agent.act([observation]).tolist()
        [observation, reward, done, info] = env.step(action)
        env.render()
        total_reward += reward
        print(observation)
        print("reward of this step:"+str(reward))
        if done or steps > 200 - 1:
            break
        print("total_reward:"+str(total_reward))


if __name__ == '__main__':
    main()


