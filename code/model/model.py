from model.value_function import *
import multiprocessing
from termcolor import *
import time

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64


class PPO(multiprocessing.Process):
    def __init__(self, args, observation_space_shape, action_space, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.args = args

    def run(self):
        self.make_model()
        # print("load learning weights")
        # self.loadModel("save_data/2018-04-09/data-01/weights/", 0)
        iteration = 0
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.args.max_kl = paths[1]
                self.task_q.task_done()
            else:
                iteration += 1
                mean_reward = self.ppo_fit(paths, iteration)
                self.task_q.task_done()
                self.result_q.put((self.get_policy(), mean_reward))
                self.saveModel(0)
                # print("Model saved")
                # if iteration > 100:
                #    self.train_writer.close()
                #    print('train end, close the log file')
        return

    def layer_norm_and_act_fn(self, layer):
        if self.args.policy_act_fn == 'selu':
            activate_func = tf.nn.selu
        else:
            activate_func = tf.nn.relu

        if self.args.policy_layer_norm:
            return activate_func(tf.contrib.layers.layer_norm(layer))
        else:
            return activate_func(layer)

    def make_model(self):
        self.epsilon = 0.2
        self.nbatch_train = self.args.policy_opt_batch
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2
        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        with tf.variable_scope("policy"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size_1, weight_init, bias_init,
                                 "policy_h1")
            h1 = self.layer_norm_and_act_fn(h1)
            h2 = fully_connected(h1, self.hidden_size_1, self.hidden_size_2, weight_init, bias_init, "policy_h2")
            h2 = self.layer_norm_and_act_fn(h2)
            h3 = fully_connected(h2, self.hidden_size_2, self.action_size, weight_init, bias_init, "policy_h3")
            h3 = tf.nn.tanh(h3)
            action_dist_logstd_param = tf.Variable((.01 * np.random.randn(1, self.action_size)).astype(np.float32),
                                                   name="policy_logstd")
        # means for each action
        self.action_dist_mu = h3
        # log standard deviations for each actions
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        batch_size = tf.shape(self.obs)[0]
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = gauss_log_prob(self.action_dist_mu, self.action_dist_logstd, self.action)
        log_oldp_n = gauss_log_prob(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)
        # importance sampling of surrogate loss (L in paper)
        self.surr = ratio * self.advantage
        self.aloss = -tf.reduce_mean(tf.minimum(
            self.surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * self.advantage))
        var_list = tf.trainable_variables()

        batch_size_float = tf.cast(batch_size, tf.float32)
        # kl divergence and shannon entropy
        self.kl = gauss_KL(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu,
                      self.action_dist_logstd) / batch_size_float
        self.ent = gauss_ent(self.action_dist_mu, self.action_dist_logstd) / batch_size_float
        # self.aloss -= 0.01 * self.ent
        self.lr = tf.Variable(1e-3, trainable=False)
        self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss, var_list=var_list)

        self.session.run(tf.global_variables_initializer())
        # value function
        self.vf = VF(self.session, self.observation_size, self.args)
        # self.vf = LinearVF()

        self.get_policy = GetPolicyWeights(self.session, var_list)
        self.saver = tf.train.Saver()

        # create logs method:
        r = tf.placeholder(tf.float32)
        r_r = tf.placeholder(tf.float32)
        kl = tf.placeholder(tf.float32)
        ent = tf.placeholder(tf.float32)
        s_loss = tf.placeholder(tf.float32)
        v_loss = tf.placeholder(tf.float32)
        tf.summary.scalar('reward', r)
        tf.summary.scalar('real_reward', r_r)
        tf.summary.scalar('kl_div', kl)
        tf.summary.scalar('entropy', ent)
        tf.summary.scalar('surrogate_loss', s_loss)
        tf.summary.scalar('value_loss', v_loss)
        # summaries merged
        merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.args.log_path + 'train', self.session.graph)

        def save_summary(summary_data, step):
            [reward, real_reward, kl_div, entropy, surrogate_loss, value_loss] = summary_data
            if reward != 0:
                feed_dict = {r: reward, r_r: real_reward, kl: kl_div, ent: entropy, s_loss: surrogate_loss, v_loss: value_loss}
                summary = self.session.run(merged, feed_dict=feed_dict)
                self.train_writer.add_summary(summary, step)

        self.save_summary = save_summary

    def extract_reward(self, paths):
        reward_list = []
        real_reward_list = []
        for path in paths:
            for episode_reward in path["episode_rewards"]:
                reward_list.append(episode_reward[0])
                real_reward_list.append(episode_reward[1])
        reward_lens = len(reward_list)
        if reward_lens:
            r, r_r = np.array(reward_list), np.array(real_reward_list)
            print(colored("%2d episode end| mean reward: %7.4f| mean real reward: %7.4f" % (reward_lens, r.mean(), r_r.mean()), "red"))
            return r.mean(), r_r.mean()
        else:
            return 0, 0

    # calculate the T time-steps advantage function A1, A2, A3, ...., AT
    def calculate_advantage(self, path):
        bs = path["obs"]
        ba = path["actions"]
        br = path["rewards"]
        bd = path["dones"]

        # print(bs.shape, ba.shape, br.shape, bd.shape)

        T = len(ba)
        gamma = self.args.gamma
        lam = self.args.lam

        last_adv = 0
        advantage = [None] * T
        discounted_return = [None] * T
        for t in reversed(range(T)):  # step T-1, T-2 ... 1
            delta = br[t] + (0 if bd[t] else gamma * self.vf.get_value(bs[t + 1])) - self.vf.get_value(bs[t])
            advantage[t] = delta + gamma * lam * (1 - bd[t]) * last_adv
            last_adv = advantage[t]

            if t == T - 1:
                discounted_return[t] = br[t] + (0 if bd[t] else gamma * self.vf.get_value(bs[t + 1]))
            else:
                discounted_return[t] = br[t] + gamma * (1 - bd[t]) * discounted_return[t + 1]
        return advantage, discounted_return

    def ppo_fit(self, paths, iteration):
        start_time = time.time()
        # is it possible to replace A(s,a) with Q(s,a)?
        for path in paths:
            path["advantage"], path["returns"] = self.calculate_advantage(path)
            # remove the last observation of the batch
            path["obs"] = path["obs"][0:-1]

        # puts all the experiences in a matrix: total_timesteps x options
        action_dist_mu = np.concatenate([path["action_dists_mu"] for path in paths])
        action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])

        eps = 1e-8
        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path["advantage"] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + eps)

        if self.args.policy_batch_fit is not True:
            feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n,
                         self.oldaction_dist_mu: action_dist_mu, self.oldaction_dist_logstd: action_dist_logstd,
                         self.lr: self.args.policy_learning_rate}
            # training with surrogate loss
            for i in range(self.args.policy_opt_epochs):
                self.session.run(self.atrain_op, feed_dict=feed_dict)
        else:
            # fit with epochs:
            nbatch = len(obs_n)
            inds = np.arange(nbatch)
            for _ in range(self.args.policy_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    feed_dict = {self.obs: obs_n[mbinds], self.action: action_n[mbinds], self.advantage: advant_n[mbinds],
                                 self.oldaction_dist_mu: action_dist_mu[mbinds],
                                 self.oldaction_dist_logstd: action_dist_logstd[mbinds],
                                 self.lr: self.args.policy_learning_rate}
                    self.session.run(self.atrain_op, feed_dict=feed_dict)

        feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n,
                     self.oldaction_dist_mu: action_dist_mu, self.oldaction_dist_logstd: action_dist_logstd}
        [surrogate_loss, kl_after, entropy_after] = self.session.run([self.aloss, self.kl, self.ent], feed_dict)

        # train value function / baseline on rollout paths
        value_loss = self.vf.fit(paths)

        fit_time = time.time() - start_time

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        episode_real_rewards = np.array([path["real_rewards"].sum() for path in paths])
        stats = {}
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Average sum of real_rewards per episode"] = episode_real_rewards.mean()
        stats["Entropy"] = entropy_after
        stats["Timesteps"] = sum([len(path["rewards"]) for path in paths])
        stats["KL between old and new distribution"] = kl_after
        stats["Surrogate loss"] = surrogate_loss
        # for k, v in stats.items():
        #    print(k + ": " + " " * (40 - len(k)) + str(v))
        # print("value function loss: ", v_loss)
        print(" ")
        print(colored("iter:{:3d}|".format(iteration), 'blue'),
              " reward:{:7.4f}| r_reward:{:7.4f}| ent:{:7.4f}| steps:{:5d}| kl:{:8.6f}".format(
              episoderewards.mean(), episode_real_rewards.mean(), entropy_after,
              sum([len(path["rewards"]) for path in paths]), kl_after))
        print("s_loss: {:8.6f}| v_loss: {:8.6f}| fit_time: {:.3f} sec|".format(surrogate_loss, value_loss, fit_time))

        mean_reward, mean_real_reward = self.extract_reward(paths)

        # if not close the train_writer, some data will lost
        self.save_summary([mean_reward, mean_real_reward, kl_after, entropy_after, surrogate_loss, value_loss], iteration)

        # return stats["Average sum of rewards per episode"]
        return mean_real_reward

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.saver.restore(self.session, save_path + "model-" + str(step))
            print("Model loaded successfully")

    def saveModel(self, step):
        if self.saver is not None:
            self.saver.save(self.session, self.args.weights_path + "model", global_step=step)

    def act(self, obs):
        action_dist_mu, action_dist_logstd = self.session.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs})
        return action_dist_mu.ravel()


class PPO1(multiprocessing.Process):
    def __init__(self, args, observation_space_shape, action_space, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.args = args

    def run(self):
        self.make_model()
        # print("load learning weights")
        # self.loadModel("save_data/2018-04-09/data-01/weights/", 0)
        iteration = 0
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.args.max_kl = paths[1]
                self.task_q.task_done()
            else:
                iteration += 1
                mean_reward = self.fit(paths, iteration)
                self.task_q.task_done()
                self.result_q.put((self.get_policy(), mean_reward))
                self.saveModel(0)
                # print("Model saved")
                # if iteration > 100:
                #    self.train_writer.close()
                #    print('train end, close the log file')
        return

    def create_policy_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            if self.args.policy_act_fn == 'selu':
                activate_func = tf.nn.selu
            else:
                activate_func = tf.nn.relu

            if self.args.policy_layer_norm:
                # two hidden layer
                l1 = tf.contrib.layers.layer_norm(
                      tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable), name='policy-l1')
                l2 = tf.contrib.layers.layer_norm(
                      tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable), name='policy-l2')
            else:
                # two hidden layer
                l1 = tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable, name='policy-l1')
                l2 = tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable, name='policy-l2')
            mu = tf.layers.dense(l2, self.action_size, tf.nn.tanh, trainable=trainable, name='policy-mu')
            sigma = tf.layers.dense(l2, self.action_size, tf.nn.softplus, trainable=trainable, name='policy-sigma')
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='policy-dist')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def create_log_method(self):
        # create logs method:
        r = tf.placeholder(tf.float32)
        r_r = tf.placeholder(tf.float32)
        kl = tf.placeholder(tf.float32)
        ent = tf.placeholder(tf.float32)
        s_loss = tf.placeholder(tf.float32)
        v_loss = tf.placeholder(tf.float32)
        tf.summary.scalar('reward', r)
        tf.summary.scalar('real_reward', r_r)
        tf.summary.scalar('kl_div', kl)
        tf.summary.scalar('entropy', ent)
        tf.summary.scalar('surrogate_loss', s_loss)
        tf.summary.scalar('value_loss', v_loss)
        # summaries merged
        merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.args.log_path + 'train', self.session.graph)

        def save_summary(summary_data, step):
            [reward, real_reward, kl_div, entropy, surrogate_loss, value_loss] = summary_data
            if reward != 0:
                feed_dict = {r: reward, r_r: real_reward, kl: kl_div, ent: entropy, s_loss: surrogate_loss,
                             v_loss: value_loss}
                summary = self.session.run(merged, feed_dict=feed_dict)
                self.train_writer.add_summary(summary, step)
        return save_summary

    def make_model(self):
        self.epsilon = 0.2
        self.nbatch_train = self.args.policy_opt_batch
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], "observation")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], "action")
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
        self.lr = tf.Variable(self.args.policy_learning_rate, trainable=False)

        # built policy network
        # new policy, will update
        pi, self.pi_params = self.create_policy_network('policy', self.obs, trainable=True)
        # old policy, don't update
        old_pi, self.old_pi_params = self.create_policy_network('old_policy', self.obs, trainable=False)

        with tf.variable_scope('entropy_pen'):
            self.entropy = tf.reduce_mean(pi.entropy())
        with tf.variable_scope('kl_divergence'):
            kl_div = tf.distributions.kl_divergence(old_pi, pi)
            self.kl = tf.reduce_mean(kl_div)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.old_pi_params)]
        with tf.variable_scope('surrogate_loss'):
            # self.ratio = pi.prob(self.action) / (old_pi.prob(self.action) + 1e-5)
            self.ratio = tf.exp(pi.log_prob(self.action) - old_pi.log_prob(self.action))
            self.surr = self.ratio * self.advantage    # surrogate loss
            self.aloss = -tf.reduce_mean(tf.minimum(
                          self.surr, tf.clip_by_value(self.ratio, 1. - self.epsilon, 1. + self.epsilon)*self.advantage))
            self.aloss -= self.args.entropy_coefficient * self.entropy
            self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss, var_list=self.pi_params)

        self.session.run(tf.global_variables_initializer())
        # value function
        self.vf = VF(self.session, self.observation_size, self.args)
        # get policy for sample actor
        self.get_policy = GetPolicyWeights(self.session, self.pi_params)
        # weights saver
        self.saver = tf.train.Saver()
        # tensorboard logs
        self.save_summary = self.create_log_method()

    def act(self, obs):
        s = np.array(obs)
        s = s[np.newaxis, :]
        a = self.session.run(self.sample_op, {self.obs: s})[0]
        return a

    # calculate the T time-steps advantage function A1, A2, A3, ...., AT
    def calculate_advantage(self, path):
        bs = path["obs"]
        ba = path["actions"]
        br = path["rewards"]
        bd = path["dones"]

        # print(bs.shape, ba.shape, br.shape, bd.shape)

        T = len(ba)
        gamma = self.args.gamma
        lam = self.args.lam

        last_adv = 0
        advantage = [None] * T
        discounted_return = [None] * T
        for t in reversed(range(T)):  # step T-1, T-2 ... 1
            delta = br[t] + (0 if bd[t] else gamma * self.vf.get_value(bs[t + 1])) - self.vf.get_value(bs[t])
            advantage[t] = delta + gamma * lam * (1 - bd[t]) * last_adv
            last_adv = advantage[t]

            if t == T - 1:
                discounted_return[t] = br[t] + (0 if bd[t] else gamma * self.vf.get_value(bs[t + 1]))
            else:
                discounted_return[t] = br[t] + gamma * (1 - bd[t]) * discounted_return[t + 1]
        return advantage, discounted_return

    def fit(self, paths, iteration):
        start_time = time.time()
        # is it possible to replace A(s,a) with Q(s,a)? i think is no...
        for path in paths:
            path["advantage"], path["returns"] = self.calculate_advantage(path)
            # remove the last observation of the batch
            path["obs"] = path["obs"][0:-1]

        # puts all the experiences in a matrix: total_timesteps x options
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])
        # obs_n, action_n = np.array(obs_n), np.array(action_n)
        # obs_n, action_n = np.vstack(obs_n), np.vstack(action_n)

        eps = 1e-8
        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path["advantage"] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + eps)
        advant_n = np.vstack(advant_n)

        # data shape:
        print("")

        # old_p0 = self.session.run(self.old_pi_params)[0][0][0:4]
        # p0 = self.session.run(self.pi_params)[0][0][0:4]
        # print("o_para 0: ", old_p0)
        # print("params 0: ", p0)

        r0 = self.session.run(self.ratio, feed_dict={self.obs: obs_n, self.action: action_n})
        surr0 = self.session.run(self.surr, feed_dict={self.obs: obs_n, self.action: action_n, self.advantage: advant_n})
        print("ratio shape: ", r0.shape)
        print("advan shape: ", advant_n.shape)
        print("surro shape: ", surr0.shape)
        print("surro 0: ", surr0[0])
        print("ratio 0: ", r0[0])
        print("adv_n 0: ", advant_n[0])

        # update the old and new policy:
        self.session.run(self.update_oldpi_op)

        if self.args.policy_batch_fit is not True:
            feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n, self.lr: self.args.policy_learning_rate}
            # training with surrogate loss
            for i in range(self.args.policy_opt_epochs):
                self.session.run(self.atrain_op, feed_dict=feed_dict)
        else:
            # fit with epochs:
            nbatch = len(obs_n)
            inds = np.arange(nbatch)
            for _ in range(self.args.policy_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    feed_dict = {self.obs: obs_n[mbinds], self.action: action_n[mbinds], self.advantage: advant_n[mbinds], self.lr: self.args.policy_learning_rate}
                    self.session.run(self.atrain_op, feed_dict=feed_dict)

        feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n}
        [surrogate_loss, kl_after, entropy_after] = self.session.run([self.aloss, self.kl, self.entropy], feed_dict)

        # train value function / baseline on rollout paths
        value_loss = self.vf.fit(paths)

        fit_time = time.time() - start_time

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        episode_real_rewards = np.array([path["real_rewards"].sum() for path in paths])
        time_steps = sum([len(path["rewards"]) for path in paths])

        print(" ")
        print(colored("*"*90, "blue"))
        print(colored("iter:{:3d}|".format(iteration), 'green'),
              " reward:{:7.4f}| r_reward:{:7.4f}| ent:{:7.4f}| steps:{:5d}| kl:{:8.6f}".format(
              episoderewards.mean(), episode_real_rewards.mean(), entropy_after, time_steps, kl_after))
        print("s_loss:{:10.8f}| v_loss:{:10.7f}| fit_time:{:.3f} sec|".format(surrogate_loss, value_loss, fit_time))

        def extract_reward(paths):
            reward_list = []
            real_reward_list = []
            for path in paths:
                for episode_reward in path["episode_rewards"]:
                    reward_list.append(episode_reward[0])
                    real_reward_list.append(episode_reward[1])
            reward_lens = len(reward_list)
            if reward_lens:
                r, r_r = np.array(reward_list), np.array(real_reward_list)
                print(colored("%2d episode end| mean reward: %7.4f| mean real reward: %7.4f" % (
                reward_lens, r.mean(), r_r.mean()), "red"))
                return r.mean(), r_r.mean()
            else:
                return 0, 0

        mean_reward, mean_real_reward = extract_reward(paths)

        # if not close the train_writer, some data will lost
        self.save_summary([mean_reward, mean_real_reward, kl_after, entropy_after, surrogate_loss, value_loss], iteration)

        # return stats["Average sum of rewards per episode"]
        return mean_real_reward

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.saver.restore(self.session, save_path + "model-" + str(step))
            print("Model loaded successfully")

    def saveModel(self, step):
        if self.saver is not None:
            self.saver.save(self.session, self.args.weights_path + "model", global_step=step)


class TRPO(multiprocessing.Process):
    def __init__(self, args, observation_space_shape, action_space, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.args = args

    def layer_norm_and_act_fn(self, layer):
        if self.args.policy_act_fn == 'selu':
            activate_func = tf.nn.selu
        else:
            activate_func = tf.nn.relu

        if self.args.policy_layer_norm:
            return activate_func(tf.contrib.layers.layer_norm(layer))
        else:
            return activate_func(layer)

    def makeModel(self):
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2
        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        bias_init = tf.constant_initializer(0)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        with tf.variable_scope("policy"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size_1, weight_init, bias_init,
                                 "policy_h1")
            h1 = self.layer_norm_and_act_fn(h1)
            h2 = fully_connected(h1, self.hidden_size_1, self.hidden_size_2, weight_init, bias_init, "policy_h2")
            h2 = self.layer_norm_and_act_fn(h2)
            h3 = fully_connected(h2, self.hidden_size_2, self.action_size, weight_init, bias_init, "policy_h3")
            h3 = tf.nn.tanh(h3)
            action_dist_logstd_param = tf.Variable((.01 * np.random.randn(1, self.action_size)).astype(np.float32),
                                                   name="policy_logstd")
        # means for each action
        self.action_dist_mu = h3
        # log standard deviations for each actions
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        batch_size = tf.shape(self.obs)[0]
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = gauss_log_prob(self.action_dist_mu, self.action_dist_logstd, self.action)
        log_oldp_n = gauss_log_prob(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)
        self.ratio = ratio
        self.log_p_n = tf.exp(log_p_n)
        self.log_oldp_n = tf.exp(log_oldp_n)
        # importance sampling of surrogate loss (L in paper)
        surr = -tf.reduce_mean(ratio * self.advantage)
        var_list = tf.trainable_variables()

        batch_size_float = tf.cast(batch_size, tf.float32)
        # kl divergence and shannon entropy
        kl = gauss_KL(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu, self.action_dist_logstd) / batch_size_float
        ent = gauss_ent(self.action_dist_mu, self.action_dist_logstd) / batch_size_float

        self.losses = [surr, kl, ent]
        # policy gradient
        self.pg = flatgrad(surr, var_list)

        # KL divergence w/ itself, with first argument kept constant.
        kl_firstfixed = gauss_selfKL_firstfixed(self.action_dist_mu, self.action_dist_logstd) / batch_size_float
        # gradient of KL w/ itself
        grads = tf.gradients(kl_firstfixed, var_list)
        # what vector we're multiplying by
        self.flat_tangent = tf.placeholder(tf.float32, [None])

        # shapes = map(var_shape, var_list)
        shapes = [self.session.run(tf.shape(v)) for v in var_list]

        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        # gradient of KL w/ itself * tangent
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        # 2nd gradient of KL w/ itself * tangent
        self.fvp = flatgrad(gvp, var_list)
        # the actual parameter values
        self.gf = GetFlat(self.session, var_list)
        # call this to set parameter values
        self.sff = SetFromFlat(self.session, var_list)
        self.session.run(tf.global_variables_initializer())
        # value function
        self.vf = VF(self.session, self.observation_size, self.args)
        # self.vf = LinearVF()

        self.get_policy = GetPolicyWeights(self.session, var_list)
        self.saver = tf.train.Saver()

        # create logs method:
        r = tf.placeholder(tf.float32)
        tf.summary.scalar('reward', r)
        # summaries merged
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.args.log_path + 'train', self.session.graph)

        def save_summary(reward, step):
            summary = self.session.run(merged, feed_dict={r: reward})
            train_writer.add_summary(summary, step)

        self.save_summary = save_summary

    def run(self):
        self.makeModel()
        # self.loadModel("save_data/2018-04-09/data-07/weights/", 0)
        iteration = 0
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.args.max_kl = paths[1]
                self.task_q.task_done()
            else:
                iteration += 1
                mean_reward = self.learn(paths, iteration)
                self.task_q.task_done()
                self.result_q.put((self.get_policy(), mean_reward))
                self.saveModel(0)
                print("Model saved")
        return

    def learn(self, paths, iteration):

        # is it possible to replace A(s,a) with Q(s,a)?
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = discount(path["rewards"], self.args.gamma)
            path["advantage"] = path["returns"] - path["baseline"]
            # path["advantage"] = path["returns"]

        # puts all the experiences in a matrix: total_timesteps x options
        action_dist_mu = np.concatenate([path["action_dists_mu"] for path in paths])
        action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])

        eps = 1e-8
        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path["advantage"] for path in paths])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + eps)

        # print("adv0: ", advant_n.mean())

        # train value function / baseline on rollout paths
        self.vf.fit(paths)

        feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n, self.oldaction_dist_mu: action_dist_mu, self.oldaction_dist_logstd: action_dist_logstd}

        # parameters
        thprev = self.gf()

        # computes fisher vector product: F * [self.pg]
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.session.run(self.fvp, feed_dict) + p * self.args.cg_damping

        g = self.session.run(self.pg, feed_dict)

        # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
        # stepdir = A_inverse * g = x
        stepdir = conjugate_gradient(fisher_vector_product, -g)

        # let stepdir =  change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        lm = np.sqrt(shs / self.args.max_kl)

        fullstep = stepdir / lm
        negative_g_dot_steppdir = -g.dot(stepdir)

        def loss(th):
            self.sff(th)
            # surrogate loss: policy gradient loss
            return self.session.run(self.losses[0], feed_dict)

        # finds best parameter by starting with a big step and working backwards
        # theta = linesearch(loss, thprev, fullstep, negative_g_dot_steppdir / lm)    # this line didn't work
        # i guess we just take a fullstep no matter what
        theta = thprev + fullstep       # theta: new weights, is equal to old weights + improvements
        self.sff(theta)

        surrogate_after, kl_after, entropy_after = self.session.run(self.losses, feed_dict)

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        stats = {}
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Entropy"] = entropy_after
        stats["max KL"] = self.args.max_kl
        stats["Timesteps"] = sum([len(path["rewards"]) for path in paths])
        stats["KL between old and new distribution"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        print('lm:', lm)
        for k, v in stats.items():
            print(k + ": " + " " * (40 - len(k)), v)  # + str(v))

        self.save_summary(stats["Average sum of rewards per episode"], iteration)

        return stats["Average sum of rewards per episode"]

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.saver.restore(self.session, save_path + "model-" + str(step))
            print("Model loaded successfully")

    def saveModel(self, step):
        if self.saver is not None:
            self.saver.save(self.session, self.args.weights_path + "model", global_step=step)

    def act(self, obs):
        action_dist_mu, action_dist_logstd = self.session.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs})
        return action_dist_mu.ravel()
