import os
import numpy as np
from keras import optimizers
from keras.layers import Dense, LSTM
from keras.models import Model
from keras.engine.input_layer import Input
from keras.preprocessing.sequence import pad_sequences

from mlp_generator import MLPSearchSpace

from CONSTANTS import *

class Controller(MLPSearchSpace):

    def __init__(self):

        self.max_len = MAX_ARCHITECTURE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM

        self.hybrid_weights = 'LOGS/hybrid_weights.h5'

        self.seq_data = []

        super().__init__(TARGET_CLASSES)

        self.controller_classes = len(self.vocab) + 1

    def sample_architecture_sequences(self, model, number_of_samples):
        final_layer_id = len(self.vocab)
        dropout_id = final_layer_id - 1
        vocab_idx = [0] + list(self.vocab.keys())
        samples = []
        print("generating architecture samples...")
        while len(samples) < number_of_samples:
            seed = []
            while len(seed) < self.max_len:
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
                sequence = sequence.reshape(1, 1, self.max_len - 1)
                (probab, _) = model.predict(sequence)
                probab = probab[0][0]
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]
                if next == dropout_id and len(seed) == 0:
                    continue
                if next == dropout_id and len(seed) == self.max_len - 2:
                    continue
                if next == final_layer_id and len(seed) == 0:
                    continue
                if next == final_layer_id:
                    seed.append(next)
                    break
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break
                if not next == 0:
                    seed.append(next)
            if seed not in self.seq_data:
                samples.append(seed)
                self.seq_data.append(seed)
        return samples

    def hybrid_control_model(self, controller_input_shape, controller_batch_size):
        main_input = Input(shape=controller_input_shape, batch_shape=controller_batch_size, name='main_input')
        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
        return model

    def train_hybrid_model(self, model, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(lr=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},
                      loss_weights={'main_output': 1, 'predictor_output': 1})
        if os.path.exists(self.hybrid_weights):
            model.load_weights(self.hybrid_weights)
        print("training controller...")
        model.fit({'main_input': x_data},
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
                   'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size)
        model.save_weights(self.hybrid_weights)

    def get_predicted_accuracies_hybrid_model(self, model, seqs):
        pred_accuracies = []
        for seq in seqs:
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies
