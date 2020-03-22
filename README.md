# Neural Architecture Search for Multi Layer Perceptrons 

Insights drawn from the following papers:

1. [ENAS](https://arxiv.org/abs/1802.03268)
2. [SeqGAN](https://arxiv.org/abs/1609.05473) 
3. [NAO](https://arxiv.org/abs/1808.07233)


## Features

The code incorporates an LSTM controller to generate sequences that represent neural network architectures, and an accuracy predictor for the generated architectures. these architectures are built into keras models, trained for certain number of epochs, evaluated, the validation accuracy being used to update the controller for better architecture search. 

1. LSTM controller with REINFORCE gradient
2. Accuracy predictor that shares weights with the above mentioned LSTM controller.
3. Weight sharing in all the architectures generated during the search phase.


## Usage

To run the architecture search:
1. Add the dataset in the datasets directory.
2. add dataset path in run.py after basic preprocessing.
3. change TARGET_CLASSES according to dataset in ```CONSTANTS.py``` 
3. run the following command from the main directory.

```bash
python3 run.py
```

To vary the search space, edit the vocab_dict() function in ```mlp_generation.py``` file. defaults mentioned below.

```python
nodes = [8,16,32,64,128,256,512]
act_funcs = ['sigmoid','tanh','relu','elu']
```

To change the NAS/controller/mlp training parameters, open the ```CONSTANTS.py``` file and edit. defaults mentioned below.

```python
########################################################
#                   NAS PARAMETERS                     #
########################################################
CONTROLLER_SAMPLING_EPOCHS = 20
SAMPLES_PER_CONTROLLER_EPOCH = 10
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 10
CONTROLLER_LOSS_ALPHA = 0.8

########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.01
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0

########################################################
#                   MLP PARAMETERS                     #
########################################################
MAX_ARCHITECTURE_LENGTH = 3
MLP_OPTIMIZER = 'Adam'
MLP_LEARNING_RATE = 0.01
MLP_DECAY = 0.0
MLP_MOMENTUM = 0.0
MLP_DROPOUT = 0.2
MLP_LOSS_FUNCTION = 'categorical_crossentropy'
MLP_ONE_SHOT = True
```

## Results

### One shot with accuracy predictor
```
Top 5 Architectures:
Architecture [(8, 'relu'), (32, 'sigmoid'), (3, 'softmax')]
Validation Accuracy: 0.7115769841454246
Architecture [(128, 'sigmoid'), (16, 'tanh'), (3, 'softmax')]
Validation Accuracy: 0.6857143044471741
Architecture [(8, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.6830426801334728
Architecture [(8, 'relu'), (512, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.6779962875626304
Architecture [(8, 'tanh'), (32, 'relu'), (3, 'softmax')]
Validation Accuracy: 0.6693877577781677
```
### One shot without accuracy predictor
```
Top 5 Architectures:
Architecture [(64, 'relu'), (32, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.7528385931795294
Architecture [(256, 'elu'), (8, 'relu'), (3, 'softmax')]
Validation Accuracy: 0.6979591846466064
Architecture [(8, 'tanh'), (16, 'tanh'), (3, 'softmax')]
Validation Accuracy: 0.6877551078796387
Architecture [(16, 'sigmoid'), (64, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.6612244844436646
Architecture [(16, 'tanh'), (512, 'sigmoid'), (3, 'softmax')]
Validation Accuracy: 0.6569202412258495
```

###  With only accuracy predictor
```
Top 5 Architectures:
Architecture [(512, 'elu'), (128, 'relu'), (3, 'softmax')]
Validation Accuracy: 0.6703154022043402
Architecture [(256, 'relu'), (128, 'relu'), (3, 'softmax')]
Validation Accuracy: 0.6697588010267778
Architecture [(256, 'elu'), (128, 'relu'), (3, 'softmax')]
Validation Accuracy: 0.669721706347032
Architecture [(16, 'sigmoid'), (8, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.6693877577781677
Architecture [(256, 'relu'), (32, 'tanh'), (3, 'softmax')]
Validation Accuracy: 0.6632652878761292
```

### Without either
``` 
Top 5 Architectures:
Architecture [(32, 'sigmoid'), (64, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.6900556434284557
Architecture [(128, 'elu'), (16, 'sigmoid'), (3, 'softmax')]
Validation Accuracy: 0.6836734414100647
Architecture [(128, 'tanh'), (512, 'sigmoid'), (3, 'softmax')]
Validation Accuracy: 0.6754360155625777
Architecture [(256, 'sigmoid'), (8, 'relu'), (3, 'softmax')]
Validation Accuracy: 0.6714285612106323
Architecture [(128, 'sigmoid'), (256, 'elu'), (3, 'softmax')]
Validation Accuracy: 0.669276423887773
```