# farm.py
# a single instance of a farm.

# the original code is wrote by qin yongliang, see his github url: https://github.com/ctmakro/stanford-osrl

# before run the agent, we should run some farms in all machines we have first,
# then create the corresponding farmlist with the address and capacity of the farms

# there will be many farms and only one farmer

# a farm should consist of a pool of instances
# and expose those instances as one giant callable class


# revised: rao zhenhuan. 2018.4.21
# info: add a sample policy to every env


import multiprocessing
import threading
import time
import random
from multiprocessing import Process, Queue
from random import randint
import traceback
import tensorflow as tf
import numpy as np
from model.utils import floatify


# num of env instance will create in the machine
ncpu = multiprocessing.cpu_count()
# the port used for communication, must be bigger than 2048
farmport = 20099   # the same in the farmer.py
        

# env wrapperenv
class RunEnv(object):
    def __init__(self, visualize=False, para_list=None):
        self.para_list = para_list
        if self.para_list["task"] == 'l2r':
            from wrapper_env.l2renv import WrapperEnv
        else:
            from wrapper_env.wrapperEnv import WrapperEnv
        print(self.para_list["task"])
        self.env = WrapperEnv(game=para_list["task"], visualize=visualize)
        self.env.seed(randint(0, 999999))
        self.observation_space_shape = self.env.observation_space_shape
        self.action_space = self.env.action_space

    def step(self, action):
        res = self.env.step(action)
        return res

    def reset(self):
        o = self.env.reset()
        return o


class Sample_policy(object):
    def __init__(self, para_list, env):
        self.env = env
        self.action_size = np.prod(self.env.action_space.shape)
        self.observation_size = self.env.observation_space_shape[0]
        self.para_list = para_list
        self.make_model()

    def make_model(self):
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        session = tf.Session(config=config)
        state = tf.placeholder(dtype=tf.float32, shape=[None, self.observation_size], name='state')
        if self.para_list["model"] == 'trpo':
            from model.policy import stochastic_gaussian_policy
            self.policy = stochastic_gaussian_policy(
                'sample_policy', session, state, self.action_size, para_list=self.para_list, trainable=False)
        elif self.para_list["model"] == 'ppo':
            if self.para_list["shared_network"]:
                from model.policy import shared_stochastic_policy_and_value
                self.policy = shared_stochastic_policy_and_value(
                    'sample_policy', session, state, self.action_size, para_list=self.para_list, trainable=False)
            else:
                from model.policy import stochastic_gaussian_policy
                self.policy = stochastic_gaussian_policy(
                    'sample_policy', session, state, self.action_size, para_list=self.para_list, trainable=False)
        else:
            from model.policy import deterministic_policy
            self.policy = deterministic_policy(
                'sample_policy', session, state, self.action_size, para_list=self.para_list, trainable=False)
            pass
        self.act = self.policy.act
        self.set_params_flat = self.policy.set_params_flat

    def rollout_an_episode(self):
        start_time = time.time()
        obs, actions, rewards, dones = [], [], [], []
        steps_time, steps, ep_time = 0, 0, 0
        o = self.env.reset()
        while True:
            before_step_time = time.time()

            o_before_a = o
            a = self.act(o_before_a)
            [o, r, d, _] = self.env.step(a)

            obs.append(floatify(o_before_a))
            actions.append(floatify(a))
            rewards.append(r)
            dones.append(d)

            step_time = time.time() - before_step_time
            steps_time += step_time
            steps += 1
            if d or steps > (self.para_list["max_pathlength"] - 1):
                ep_time = time.time() - start_time
                obs.append(floatify(o))
                break
        return [obs, rewards, dones, actions, ep_time, steps_time / steps]


# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def standalone_headless_isolated(pq, cq, plock, para_list=None):
    # locking to prevent mixed-up printing.
    plock.acquire()
    print('starting headless...', pq, cq)
    try:
        env = RunEnv(visualize=False, para_list=para_list)
        policy = Sample_policy(para_list, env)
    except Exception as e:
        print('error on start of standalone')
        traceback.print_exc()
        plock.release()
        return
    else:
        plock.release()

    def report(e):
        # a way to report errors ( since you can't just throw them over a pipe )
        # e should be a string
        print('(standalone) got error!!!')
        cq.put(('error', e))

    try:
        while True:
            # messages should be tuples,
            # msg[0] should be string
            msg = pq.get()

            if msg[0] == 'reset':
                o = env.reset()
                cq.put(floatify(o))
            elif msg[0] == 'step':
                # msg[1]: action
                o, r, d, i = env.step(msg[1])
                o = floatify(o)
                cq.put((o, r, d, i))
            elif msg[0] == 'set':
                # msg[1]: policy parameters
                policy.set_params_flat(msg[1])
                cq.put(('success', [1]))
            elif msg[0] == 'reset_and_rollout':
                path = policy.rollout_an_episode()
                cq.put(path)
            else:
                cq.close()
                pq.close()
                del env
                break
    except Exception as e:
        traceback.print_exc()
        report(str(e))

    return  # end process


# global process lock
plock = multiprocessing.Lock()
# global thread lock
tlock = threading.Lock()
# global id issurance
eid = int(random.random()*100000)


def get_eid():
    global eid, tlock
    tlock.acquire()
    i = eid
    eid += 1
    tlock.release()
    return i


# class that manages the interprocess communication and expose itself as a RunEnv.
# reinforced: this class should be long-running. it should reload the process on errors.
class ei:  # Environment Instance
    def __init__(self, para_list=None):
        self.para_list = para_list
        self.occupied = False   # is this instance occupied by a remote client
        self.id = get_eid()     # what is the id of this environment
        self.pretty('instance creating')

        self.newproc()
        self.lock = threading.Lock()

    def timer_update(self):
        self.last_interaction = time.time()

    def is_occupied(self):
        if self.occupied is False:
            return False
        else:
            if time.time() - self.last_interaction > 20*60:
                # if no interaction for more than 20 minutes
                self.pretty('no interaction for too long, self-releasing now. applying for a new id.')
                self.id = get_eid()   # apply for a new id.
                self.occupied = False
                self.pretty('self-released.')
                return False
            else:
                return True

    def occupy(self):
        self.lock.acquire()
        if self.is_occupied() is False:
            self.occupied = True
            self.id = get_eid()
            self.lock.release()
            return True    # on success
        else:
            self.lock.release()
            return False   # failed

    def release(self):
        self.lock.acquire()
        self.occupied = False
        self.id = get_eid()
        self.lock.release()

    # create a new RunEnv in a new process.
    def newproc(self):
        global plock
        self.timer_update()

        self.pq, self.cq = Queue(1), Queue(1)   # two queue needed

        self.p = Process(
            target=standalone_headless_isolated,
            args=(self.pq, self.cq, plock, self.para_list)
        )
        self.p.daemon = True    # if daemon is ture, the sub Process will end if the father Process end
        self.reset_count = 0    # how many times has this instance been reset() ed
        self.step_count = 0
        self.timer_update()
        self.p.start()
        return

    # send x to the process
    def send(self, x):
        return self.pq.put(x)

    # receive from the process.
    def recv(self):
        # receive and detect if we got any errors
        r = self.cq.get()

        # isinstance is dangerous, commented out
        # if isinstance(r, tuple):
        if r[0] == 'error':
            # read the exception string
            e = r[1]
            self.pretty('got exception')
            self.pretty(e)
            raise Exception(e)
        return r

    def reset_and_rollout(self):
        self.timer_update()
        if not self.is_alive():
            # if our process is dead for some reason
            self.pretty('process found dead on reset(). reloading.')
            self.kill()
            self.newproc()

        if self.reset_count > 50:  # if resetted for more than 50 times
            self.pretty('environment has been resetted too much. '
                        'memory leaks and other problems might present. reloading.')
            self.kill()
            self.newproc()

        self.reset_count += 1
        self.send(('reset_and_rollout',))
        r = self.recv()
        self.timer_update()
        return r

    def reset(self):
        self.timer_update()
        if not self.is_alive():
            # if our process is dead for some reason
            self.pretty('process found dead on reset(). reloading.')
            self.kill()
            self.newproc()

        if self.reset_count > 50 or self.step_count > 10000:  # if resetted for more than 50 times
            self.pretty('environment has been resetted too much. '
                        'memory leaks and other problems might present. reloading.')
            self.kill()
            self.newproc()

        self.reset_count += 1
        self.send(('reset',))
        r = self.recv()
        self.timer_update()
        return r

    def step(self, actions):
        self.timer_update()
        self.send(('step', actions,))
        r = self.recv()
        self.timer_update()
        self.step_count += 1
        return r

    def set(self, theta):
        self.timer_update()
        self.send(('set', theta))
        r = self.recv()
        self.timer_update()
        return r

    def kill(self):
        if not self.is_alive():
            self.pretty('process already dead, no need for kill.')
        else:
            self.send(('exit',))
            self.pretty('waiting for join()...')

            while 1:
                self.p.join(timeout=5)
                if not self.is_alive():
                    break
                else:
                    self.pretty('process is not joining after 5s, still waiting...')
            self.pretty('process joined.')

    def __del__(self):
        self.pretty('__del__')
        self.kill()
        self.pretty('__del__ accomplished.')

    def is_alive(self):
        return self.p.is_alive()

    # pretty printing
    def pretty(self, s):
        if self.para_list["farm_debug_print"]:
            print(('(ei) {} ').format(self.id) + str(s))


# class that other classes acquires and releases EIs from.
class eipool:  # Environment Instance Pool
    def __init__(self, para_list=None, n=1):
        self.para_list = para_list
        self.pretty('starting ' + str(n) + ' instance(s)...')
        self.pool = [ei(self.para_list) for i in range(n)]
        self.lock = threading.Lock()

    def acq_env(self):
        self.lock.acquire()
        for e in self.pool:
            if e.occupy() is True:  # successfully occupied an environment
                self.lock.release()
                return e   # return the envinstance

        self.lock.release()
        return False  # no available ei

    def rel_env(self, ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.release()  # freed
        self.lock.release()

    # def num_free(self):
    #     return sum([0 if e.is_occupied() else 1 for e in self.pool])
    #
    # def num_total(self):
    #     return len(self.pool)
    #
    # def all_free(self):
    #     return self.num_free()==self.num_total()

    def get_env_by_id(self, id):
        for e in self.pool:
            if e.id == id:
                return e
        return False

    def pretty(self, s):
        if self.para_list["farm_debug_print"]:
            print(('(eipool) ') + str(s))

    def __del__(self):
        for e in self.pool:
            del e


# farm
# interface with eipool via eids.
# ! this class is a singleton. must be made thread-safe.
class farm:
    def __init__(self):
        self.lock = threading.Lock()
        self.para_list = {}

    def get_para_list(self, para_list=None):
        self.lock.acquire()
        self.para_list = para_list
        self.pretty('get the parameter_list with the model')
        for k, v in self.para_list.items():
            print(('(farm) ') + k + ": " + str(v))
        self.lock.release()

    def acq(self, n=None):
        self.renew_if_needed(n)
        result = self.eip.acq_env()     # thread-safe
        if result is False:
            ret = False
        else:
            self.pretty('acq ' + str(result.id))
            ret = result.id
        return ret

    def rel(self, id):
        e = self.eip.get_env_by_id(id)
        if e is False:
            self.pretty(str(id) + ' not found on rel(), might already be released')
        else:
            self.eip.rel_env(e)
            self.pretty('rel ' + str(id))

    def step(self, id, actions):
        e = self.eip.get_env_by_id(id)
        if e is False:
            self.pretty(str(id) + ' not found on step(), might already be released')
            return False

        try:
            ordi = e.step(actions)
            return ordi
        except Exception as e:
            traceback.print_exc()
            raise e

    def reset(self, id):
        e = self.eip.get_env_by_id(id)
        if e is False:
            self.pretty(str(id) + ' not found on reset(), might already be released')
            return False

        try:
            oo = e.reset()
            return oo
        except Exception as e:
            traceback.print_exc()
            raise e

    def set(self, id, theta):
        e = self.eip.get_env_by_id(id)
        if e is False:
            self.pretty(str(id) + ' not found on reset(), might already be released')
            return False

        try:
            flag = e.set(theta)
            return flag
        except Exception as e:
            traceback.print_exc()
            raise e

    def reset_and_rollout(self, id):
        e = self.eip.get_env_by_id(id)
        if e is False:
            self.pretty(str(id) + ' not found on reset(), might already be released')
            return False

        try:
            [obs, rewards, dones, actions, ep_time, step_time] = e.reset_and_rollout()
            return [obs, rewards, dones, actions, ep_time, step_time]
        except Exception as e:
            traceback.print_exc()
            raise e

    def renew_if_needed(self, n=None):
        self.lock.acquire()
        if not hasattr(self, 'eip'):
            self.pretty('renew because no eipool present')
            self._new(n)
        self.lock.release()

    def forcerenew(self, n=None):
        self.lock.acquire()
        self.pretty('forced pool renew')

        if hasattr(self, 'eip'):   # if eip exists
            del self.eip
        self._new(n)
        self.lock.release()

    def _new(self, n=None):
        e_num = ncpu if n is None else n
        self.eip = eipool(self.para_list, e_num)

    def pretty(self, s):
        if self.para_list["farm_debug_print"]:
            print(('(farm) ') + str(s))


# expose the farm via Pyro4
def main():
    from remote_env.pyro_helper import pyro_expose
    pyro_expose(farm, farmport, 'farm')


if __name__ == '__main__':
    main()
