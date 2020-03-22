import pandas as pd

from utils import *
from mlpnas import MLPNAS
from CONSTANTS import TOP_N


data = pd.read_csv('DATASETS/wine-quality.csv')
x = data.drop('quality_label', axis=1, inplace=False).values
y = pd.get_dummies(data['quality_label']).values

nas_object = MLPNAS(x, y)
data = nas_object.search()

log_event()

get_top_n_architectures(TOP_N)
