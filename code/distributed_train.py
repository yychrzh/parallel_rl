from tools import save_list, create_save_path, create_path
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import numpy as np


HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64


"""a single version ddpg algorithm """
STATE_FLAG = 'debug'      # choose between 'debug', 'train' and 'test' to save the data to different paths


if __name__ == '__main__':
    # create a sub folder in current path with time index
    main_save_path = create_save_path('../' + STATE_FLAG + '_data/')
    print("create main_save_path ", main_save_path, " in current path success")

    # parameter list
    param_list = [
        ["single"],
        ["--model",                str('trpo')],
        ["--task",                 str('Pendulum-v0')],
        ["--debug_print",          bool(True)],
        ["--weights_path",         str(main_save_path + 'weights/')],
        ["--log_path",             str(main_save_path + 'logs/')],
        ["--memory_path",          str(main_save_path + 'memory/')],
        ["--list_path",            str(main_save_path + 'save_list/')],
        ["--timesteps_per_batch",  int(5000)],
        ["--n_steps",              int(100000)],
        ["--n_episodes",           int(10000)],
        ["--gamma",                float(0.95)],
        ["--lam",                  float(0.95)],
        ["--policy_opt_epochs",    int(15)],
        ["--value_opt_epochs",     int(30)],
        ["--policy_opt_batch",     int(64)],
        ["--value_opt_batch",      int(64)],
        ["--policy_learning_rate", float(3e-4)],
        ["--value_learning_rate",  float(3e-4)],
        ["--policy_batch_fit",     bool(True)],
        ["--value_batch_fit",      bool(True)],
        ["--policy_layer_norm",    bool(False)],
        ["--value_layer_norm",     bool(False)],
        ["--policy_act_fn",        str('relu')],
        ["--value_act_fn",         str('relu')],
        ["--network_shape",        list([HIDDEN_SIZE_1, HIDDEN_SIZE_2])],
        ["--value_coefficient",    float(0.1)],
        ["--entropy_coefficient",  float(0.00)],
        ["--sample_length",        int(400)],
        ["--max_kl",               float(0.1)],
        ["--cg_damping",           float(1e-3)],
        ["--num_threads",          int(1)],
        ["--monitor",              bool(False)],
        ["--max_pathlength",       int(200)],
        ["--collect_every_step",   bool(False)],
        ["--farmer_port",          int(20099)],
        ["--farmer_debug_print",   bool(False)],
        ["--farm_debug_print",     bool(False)],
        ["--farm_list_base",       list([('127.0.0.1', 5)])]
    ]

    parser = argparse.ArgumentParser(description=param_list[0][0])
    for i in range(len(param_list)):
        if i > 0:
            parser.add_argument(param_list[i][0], type=type(param_list[i][1]), default=param_list[i][1])

    args = parser.parse_args()
    print("create argparse success")

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_info = [['time: ', current_time], ['param_info: ', param_list]]
    save_list(save_info, 'create_info', main_save_path + 'create_info')
    # create weights path
    create_path(args.weights_path)
    # create memory path
    create_path(args.memory_path)
    # create log path
    create_path(args.log_path)
    # create save_list path
    create_path(args.list_path)
    print("create save path success")
    history_reward = []

    # import env:
    if args.task == 'l2r':
        from wrapper_env.l2renv import WrapperEnv
    else:
        from wrapper_env.wrapperEnv import WrapperEnv
    env = WrapperEnv(game=args.task)

    # import agent and rollouts method:
    if args.model == 'ddpg':
        from model.ddpg import DDPG as Agent
        from rollouts.rollouts import off_policy_parallel_rollouts as rollouts  # should run farm.py first
    elif args.model == 'ppo':
        from model.ppo import PPO as Agent
        from rollouts.rollouts import on_policy_parallel_rollouts as rollouts  # should run farm.py first
    else:
        from model.trpo import TRPO as Agent
        from rollouts.rollouts import on_policy_parallel_rollouts as rollouts  # should run farm.py first

    agent = Agent(args, env.observation_space_shape, env.action_space)

    runner = rollouts(agent, env=env)
    history_reward = runner.rollout()

    # plot the history reward after the train end
    plot_list = []
    for i in range(len(history_reward)):
        if i == 0:
            plot_list.append(history_reward[i])
        else:
            plot_list.append(history_reward[i - 1] * 0.9 + history_reward[i] * 0.1)
    # plot the mean_reward
    plt.plot(np.arange(len(plot_list)), plot_list)
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.savefig(main_save_path + "fig1.jpg")
    plt.show()