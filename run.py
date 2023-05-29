import pandas as pd
import tensorflow as tf

from CONSTANTS import TOP_N, RANDOM_SEED

from numpy.random import seed

seed(RANDOM_SEED)
tf.compat.v2.random.set_seed(RANDOM_SEED)

from utils import *
from mlpnas import MLPNAS

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
#print(nas_object.controller_model.summary())
data = nas_object.search()

get_top_n_architectures(TOP_N)

# data = load_nas_data()
# data = sort_search_data(data)
# for seq_data in data[:TOP_N]:
#     print('Model')
#     model = nas_object.create_architecture(seq_data[0])
#     print(model.summary())
#     print("Evaluate inference time cost...")
#     latency_results = nas_object.evaluate_latency(model)
#     print(latency_results)
#     nas_object.load_shared_weights(model)
#     results = nas_object.inference_architecture(model)
#     print("test loss, test acc:", results)
