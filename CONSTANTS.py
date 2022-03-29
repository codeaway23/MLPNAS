########################################################
#                   NAS PARAMETERS                     #
########################################################
CONTROLLER_SAMPLING_EPOCHS = 1000
SAMPLES_PER_CONTROLLER_EPOCH = 10
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 5
CONTROLLER_LOSS_ALPHA = 0.9

########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.01
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0
CONTROLLER_USE_PREDICTOR = True

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

########################################################
#                   DATA PARAMETERS                    #
########################################################
TARGET_CLASSES = 3

########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = 5

########################################################
#                    RANDOM SEED                       #
########################################################

RANDOM_SEED = 1