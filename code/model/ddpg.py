import threading as th
import time
from model.rpm import rpm
from model.noise import one_fsq_noise
from model.policy import deterministic_policy
from model.value_function import q_value_function
from model.utils import *


# deep deterministic policy gradient


class DDPG(object):
    def __init__(self, para_list, observation_space_shape, action_space):
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.para_list = para_list
        self.rpm = rpm(self.para_list["replay_memory_size"])
        self.training = True
        self.noise_source = one_fsq_noise()
        self.lock = th.Lock()     # to make the agent thread safe with distributed training
        self.make_model()

    def create_actor_train_method(self):
        self.actor_lr = tf.Variable(self.para_list["policy_learning_rate"], trainable=False)
        self.q_gradient_input = tf.placeholder(tf.float32, [None, self.action_size], 'q_gradient')
        self.actor_parameters_gradients = tf.gradients(self.actor.policy, self.actor.params, -self.q_gradient_input)
        self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(
            zip(self.actor_parameters_gradients, self.actor.params))

    def create_soft_update_method(self):
        self.get_actor_flat = GetFlat(self.session, self.actor.params)
        self.get_critic_flat = GetFlat(self.session, self.critic.params)
        self.set_target_actor_flat = SetFromFlat(self.session, self.target_actor.params)
        self.set_target_critic_flat = SetFromFlat(self.session, self.target_critic.params)
        self.get_from_flat = self.get_actor_flat

    # tau: soft update rat: the Recommended value in the paper is 1e-3
    def update_target_network(self, tau):
        start_time = time.time()
        self.lock.acquire()
        # update target actor network from actor network:
        actor_params_flat = self.get_actor_flat()
        self.set_target_actor_flat(actor_params_flat, tau)
        # update target critic network from critic network:
        critic_params_flat = self.get_critic_flat()
        self.set_target_critic_flat(critic_params_flat, tau)
        self.lock.release()
        cost_time = time.time() - start_time
        return cost_time

    def make_model(self):
        self.soft_update_rate = 1e-3
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], "observation")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], "action")

        self.actor = deterministic_policy(
            'actor', self.session, self.obs, self.action_size, self.para_list, trainable=True)
        self.target_actor = deterministic_policy(
            'target_actor', self.session, self.obs, self.action_size, self.para_list, trainable=False)
        self.critic = q_value_function('critic', self.session, self.observation_size,
                                       self.action_size, self.para_list, trainable=True)
        self.target_critic = q_value_function('target_critic', self.session, self.observation_size,
                                       self.action_size, self.para_list, trainable=False)

        self.create_actor_train_method()
        self.create_soft_update_method()

        self.session.run(tf.global_variables_initializer())

        # predict action and value
        self.act = self.actor.act
        self.noise_act = self.actor.noise_act
        self.act_batch = self.actor.act_batch
        self.target_act_batch = self.target_actor.act_batch

        # weights saver
        self.saver = tf.train.Saver()

        # copy params from network to target network
        self.update_target_network(1.0)

    def update_actor_network(self, batch_samples, learning_rate=1e-3):
        start_time = time.time()
        [s1, _, _, _, _] = batch_samples
        a_predict = self.act_batch(s1)  # predict actions for s1
        q_gradient_batch = self.critic.value_action_gradients(s1, a_predict)
        self.lock.acquire()
        feed_dict = {self.obs: s1, self.q_gradient_input: q_gradient_batch, self.actor_lr: learning_rate}
        self.session.run(self.actor_train_op, feed_dict=feed_dict)
        self.lock.release()
        q_predict = self.critic.value_batch(s1, a_predict)
        actor_loss = -np.concatenate(q_predict).mean()
        # the under code is bug with too much calculate time consumed!
        # self.session.run(tf.reduce_mean(-self.critic), feed_dict={self.obs: s1, self.action: a_predict})
        cost_time = time.time() - start_time
        return actor_loss, cost_time

    def feed_one(self, tup):
        self.rpm.add(tup)  # the add method itself is thread safe

    # sample a batch samples from rpm with batch_size
    def sample_one(self, batch_size):
        start_time = time.time()
        samples = self.rpm.sample_batch(batch_size)
        cost_time = time.time() - start_time
        return samples, cost_time

    def fit(self, learning_rate=(1e-3, 1e-3)):
        batch_size = self.para_list["policy_opt_batch"]
        opt_times = self.para_list["policy_opt_epochs"]
        actor_loss, critic_loss = 0, 0
        mean_s_time, mean_a_time, mean_c_time, mean_u_time = 0, 0, 0, 0
        start_time = time.time()
        if self.training and self.rpm.size() > batch_size * 32:
            for i in range(opt_times):
                samples, s_cost_time = self.sample_one(batch_size)
                a_loss, a_cost_time = self.update_actor_network(samples, learning_rate=learning_rate[0])
                y_target = calculate_y_target(samples, self.target_actor, self.target_critic, self.para_list["gamma"])
                s1 = samples[0]
                a1 = samples[1]
                c_loss, c_cost_time = self.critic.fit([s1, a1, y_target], learning_rate=learning_rate[1])
                u_cost_time = self.update_target_network(self.soft_update_rate)

                actor_loss += a_loss / opt_times
                critic_loss += c_loss / opt_times
                mean_s_time += s_cost_time / opt_times
                mean_a_time += a_cost_time / opt_times
                mean_c_time += c_cost_time / opt_times
                mean_u_time += u_cost_time / opt_times
        fit_cost_time = time.time() - start_time
        return actor_loss, critic_loss, fit_cost_time, mean_s_time, mean_a_time, mean_c_time, mean_u_time

    def saveModel(self, step):
        if self.saver is not None:
            self.lock.acquire()
            self.saver.save(self.session, self.para_list["weights_path"] + "model", global_step=step)
            self.lock.release()

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.lock.acquire()
            self.saver.restore(self.session, save_path + "model-" + str(step))
            self.lock.release()
            print("Model loaded successfully")