import os
import shutil
import pickle
import numpy as np
from CONSTANTS import *
from mlp_generator import MLPSearchSpace

########################################################
#                   DATA PROCESSING                    #
########################################################


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

########################################################
#                       LOGGING                        #
########################################################


def clean_log():
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))


def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)


def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))

########################################################
#                     EVALUATION                       #
########################################################


def get_top_n_architectures(n, event):
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)
    with open(data_file, 'r') as f:
        data = pickle.load(f)
    search_space = MLPSearchSpace(TARGET_CLASSES)
    print('Top {} Architectures:').format(n)
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])

