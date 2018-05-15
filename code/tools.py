import json
import os
import time


def is_path_exists(path):
    isExists = os.path.exists(path)
    if not isExists:
        return 0
    else:
        return 1


# create new path in current path
def create_path(path):
    abs_path = os.path.abspath('.')
    save_path = os.path.join(abs_path, path)
    if not is_path_exists(save_path):
        os.makedirs(save_path)


# save a list object to a file_name.txt file in save_path in current path
def save_list(data_list, file_name, save_path):
    json_str = json.dumps(data_list)
    create_path(save_path)
    file = open(save_path + '/' + file_name + '.txt', 'w')
    for d_list in json_str:
        file.write(d_list)
    file.close()


# load the list object from file_name.txt in save_path in current path
def load_list(file_name, path):
    start_time = time.time()
    file = open(path + '/' + file_name + '.txt', 'r')
    json_str = file.read()
    data_list = json.loads(json_str)
    if data_list is not None:
        total_time = time.time() - start_time
        print('load success, cost %.4f sec, close the file' % (total_time))
    else:
        print('load failed, close the file')
    file.close()
    return data_list


# create a sub path with the index of time in current path
def create_save_path(first_path):
    abs_path = os.path.abspath('.')

    # add date info to the save_path
    folder = ''
    for i in range(3):
        if time.localtime()[i] < 10:
            folder += '0'
        folder += str(time.localtime()[i])
        if i < 2:
            folder += '-'

    # add step index to the save_path
    step = 1
    save_path = first_path + folder + '/data-' + '0' + str(step) + '/'
    while is_path_exists(os.path.join(abs_path, save_path)):
        step = int(step) + 1
        if step < 10:
            save_path = first_path + folder + '/data-' + '0' + str(step) + '/'
        else:
            save_path = first_path + folder + '/data-' + str(step) + '/'
    return save_path



