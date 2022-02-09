import pandas as pd
import tensorflow as tf

from utils import *
from mlpnas import MLPNAS
from CONSTANTS import TOP_N

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

data = pd.read_csv('DATASETS/wine-quality.csv')
x = data.drop('quality_label', axis=1, inplace=False).values
y = pd.get_dummies(data['quality_label']).values

nas_object = MLPNAS(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)
