# farmer.py

# the original code is wrote by qin yongliang, see his github url: https://github.com/ctmakro/stanford-osrl

# connector to the farms, return the remote object


from remote_env.pyro_helper import pyro_connect
import time


# a farm of 4 cores is available on localhost, while a farm of 8 available maybe on another machine.
# expand the list if you have more machines.
# this file will be consumed by the host to find the slaves.


class farmlist:
    def __init__(self, farm_port):
        self.list = []
        self.farm_port = farm_port

    def generate(self):

        def addressify(farmaddr, port):
            return farmaddr + ':' + str(port)

        addresses = [addressify(farm[0], self.farm_port) for farm in self.list]
        capacities = [farm[1] for farm in self.list]
        failures = [0 for i in range(len(capacities))]

        return addresses, capacities, failures

    def push(self, addr, capa):
        self.list.append((addr, capa))


class remoteEnv:
    def __init__(self, fp, id, debug_print):  # fp = farm proxy
        self.fp = fp
        self.id = id
        self.debug_print = debug_print

    def reset_and_rollout(self):
        return self.fp.reset_and_rollout(self.id)

    def reset(self):
        return self.fp.reset(self.id)

    def step(self, actions):
        ret = self.fp.step(self.id, actions)
        if ret is False:
            self.pretty('env not found on farm side, might been released.')
            raise Exception('env not found on farm side, might been released.')
        return ret
    
    def set(self, theta):
        ret = self.fp.set(self.id, theta)
        if ret[0] != 'success':
            self.pretty('sample policy set wrong.')
            raise Exception('sample policy set wrong.')
        return ret

    def rel(self):
        count = 0
        while True:  # releasing is important, so
            try:
                count += 1
                self.fp.rel(self.id)
                break
            except Exception as e:
                self.pretty('exception caught on rel()')
                self.pretty(e)
                time.sleep(3)
                if count > 5:
                    self.pretty('failed to rel().')
                    break
                pass

        self.fp._pyroRelease()

    def pretty(self, s):
        if self.debug_print:
            print(('(remoteEnv) {} ').format(self.id) + str(s))

    def __del__(self):
        self.rel()


class farmer:
    def __init__(self, para_list=None):
        self.addresses, self.capacities, self.failures = [], [], []
        self.para_list = para_list
        self.fl = farmlist(self.para_list["farmer_port"])
        self.reload_addr()

        for idx, address in enumerate(self.addresses):

            fp = pyro_connect(address, 'farm')
            try:
                self.pretty('fp.get_para_list() success on ' + address)
                fp.get_para_list(self.para_list)
            except Exception as e:
                self.pretty('fp.get_para_list() failed on ' + address)
                self.pretty(e)
                fp._pyroRelease()
                continue

            fp = pyro_connect(address, 'farm')
            self.pretty('forced renewing... ' + address)
            try:
                fp.forcerenew(self.capacities[idx])
                self.pretty('fp.forcerenew() success on ' + address)
            except Exception as e:
                self.pretty('fp.forcerenew() failed on ' + address)
                self.pretty(e)
                fp._pyroRelease()
                continue
            fp._pyroRelease()

    # find non-occupied instances from all available farms
    def acq_env(self):
        ret = False

        import random   # randomly sample to achieve load averaging
        l = list(range(len(self.addresses)))
        random.shuffle(l)

        for idx in l:
            time.sleep(0.1)
            address = self.addresses[idx]
            capacity = self.capacities[idx]

            if self.failures[idx] > 0:
                # wait for a few more rounds upon failure,
                # to minimize overhead on querying busy instances
                self.failures[idx] -= 1
                continue
            else:
                fp = pyro_connect(address, 'farm')
                # fp.get_para_list(self.para_list)
                try:
                    result = fp.acq(capacity)
                except Exception as e:
                    self.pretty('fp.acq() failed on ' + address)
                    self.pretty(e)

                    fp._pyroRelease()
                    self.failures[idx] += 4
                    continue
                else:
                    if result is False:  # no free ei
                        fp._pyroRelease()  # destroy proxy
                        self.failures[idx] += 4
                        continue
                    else:     # result is an id
                        eid = result
                        # build remoteEnv around the proxy
                        renv = remoteEnv(fp, eid, self.para_list["farmer_debug_print"])
                        self.pretty('got one on {} id:{}'.format(address, eid))
                        ret = renv
                        break

        # ret is False if none of the farms has free ei
        return ret

    def reload_addr(self):
        self.pretty('reloading farm list...')
        self.fl.list = []
        for item in self.para_list["farm_list_base"]:
            self.fl.push(item[0], item[1])

        self.addresses, self.capacities, self.failures = self.fl.generate()

    def pretty(self, s):
        if self.para_list["farmer_debug_print"]:
            print('(farmer) ' + str(s))
