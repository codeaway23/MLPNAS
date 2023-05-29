import pickle
import sys
import tvm
from tvm import relay
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
import tensorflow as tf
import keras
import numpy as np
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from CONSTANTS import *
from controller import Controller
from mlp_generator import MLPGenerator

from utils import *


class MLPNAS(Controller):

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        self.data = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        clean_log()

        super().__init__()

        self.model_generator = MLPGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (MAX_ARCHITECTURE_LENGTH,
                                       self.controller_classes)
        if self.use_predictor:
            self.controller_model = self.hybrid_control_model(
                self.controller_input_shape, self.controller_batch_size)
        else:
            self.controller_model = self.control_model(
                self.controller_input_shape, self.controller_batch_size)

        self.target_dev = "llvm -mcpu=core-avx2"
        #self.target_dev = "cuda"
        #self.target_dev = "cuda -libs=cudnn"
        self.tvm_opt_level = 3
        self.tvm_module = ''

    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.model_generator.loss_func = 'binary_crossentropy'
        model = self.model_generator.create_model(sequence,
                                                  np.shape(self.x[0]))
        model = self.model_generator.compile_model(model)
        return model

    def train_architecture(self, model):
        x, y = unison_shuffled_copies(self.x, self.y)
        history = self.model_generator.train_model(
            model, x, y, self.architecture_train_epochs)
        return history

    def load_shared_weights(self, model):
        self.model_generator.load_shared_weights(model)

    def to_tvm_module(self, model):
        shape_dict = {model.input_names[0]: (1, self.x.shape[1])}
        mod, params = relay.frontend.from_keras(model, shape_dict)

        target = tvm.target.Target(self.target_dev)
        dev = tvm.device(str(target), 0)

        with tvm.transform.PassContext(opt_level=self.tvm_opt_level):
            lib = relay.build_module.build(mod, target=target, params=params)
        module = graph_executor.GraphModule(lib["default"](dev))
        data_tvm = tvm.nd.array(
            (np.random.uniform(size=(1, self.x.shape[1]))).astype("float32"))
        module.set_input(model.input_names[0], data_tvm)
        return module, dev

    def evaluate_latency(self, model):
        module, dev = self.to_tvm_module(model)
        return module.benchmark(dev, repeat=10, min_repeat_ms=500)

    def inference_architecture(self, model):
        results = self.model_generator.inference_model(model, self.x, self.y)
        return results

    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        if len(history.history['val_accuracy']) == 1:
            if pred_accuracy:
                self.data.append([
                    sequence, history.history['val_accuracy'][0], pred_accuracy
                ])
                print('predict accuracy: ', pred_accuracy, ' (',
                      pred_accuracy - history.history['val_accuracy'][0], ')')
            else:
                self.data.append(
                    [sequence, history.history['val_accuracy'][0]])
            print('validation accuracy: ', history.history['val_accuracy'][0])
        else:
            val_acc = np.ma.average(
                history.history['val_accuracy'],
                weights=np.arange(1,
                                  len(history.history['val_accuracy']) + 1),
                axis=-1)
            if pred_accuracy:
                self.data.append([sequence, val_acc, pred_accuracy])
                print('predict accuracy: ', pred_accuracy, ' (',
                      pred_accuracy - val_acc, ')')
            else:
                self.data.append([sequence, val_acc])
            print('validation accuracy: ', val_acc)

    def prepare_controller_data(self, sequences):
        final_layer_id = len(self.vocab)
        # pad with 'start' token
        controller_sequences = pad_sequences(sequences,
                                             maxlen=self.max_len,
                                             padding='post',
                                             value=final_layer_id)
        controller_sequences = pad_sequences(sequences,
                                             maxlen=self.max_len + 1,
                                             padding='pre',
                                             value=0)
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences),
                                                  self.max_len, 1)
        xc = to_categorical(xc, self.controller_classes)
        yc = controller_sequences[:, 1:].reshape(len(controller_sequences),
                                                 self.max_len, 1)
        val_acc_target = []
        for idx in range(len(self.data)):
            data_arch = self.data[idx][0]
            data_arch = np.pad(data_arch, (0, self.max_len - len(data_arch)),
                               constant_values=final_layer_id)
            data_acc = np.ones([self.max_len],
                               dtype=np.float) * self.data[idx][1]
            data_acc = np.where(data_arch == final_layer_id, data_acc, 0.0)
            val_acc_target.append(data_acc)
        val_acc_target = np.array(val_acc_target)
        return xc, yc, val_acc_target

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r -
                        discounted_r.mean()) / discounted_r.std()
        return discounted_r

    def custom_loss(self, target, output):
        reward = np.array([
            item[1] for item in self.data[-self.samples_per_controller_epoch:]
        ]).reshape(self.samples_per_controller_epoch, 1)
        reward_norm = (reward - reward.mean()) / (reward.std())

        discounted_reward = self.get_discounted_reward(reward_norm)
        # select action probability for each layer from target
        # and get action probability for each architecture with
        # conditional probability
        sel_output = tf.gather(output, target, axis=2)
        sel_output = tf.math.reduce_prod(sel_output, axis=1)
        loss = -K.log(sel_output) * discounted_reward[:, None]
        return loss

    def train_controller(self, model, x, y, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model, x, y, pred_accuracy,
                                    self.custom_loss, len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model, x, y, self.custom_loss,
                                     len(self.data),
                                     self.controller_train_epochs)

    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print(
                '------------------------------------------------------------------'
            )
            print('                       CONTROLLER EPOCH: {}'.format(
                controller_epoch))
            print(
                '------------------------------------------------------------------'
            )
            sequences = self.sample_architecture_sequences(
                self.controller_model, self.samples_per_controller_epoch)
            sys.stdout.flush()
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(
                    self.controller_model, sequences)
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence),
                      sequence)
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                if self.use_predictor:
                    self.append_model_metrics(sequence, history,
                                              pred_accuracies[i][0])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
                sys.stdout.flush()
            xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            self.train_controller(
                self.controller_model, xc, yc,
                val_acc_target[-self.samples_per_controller_epoch:])
            sys.stdout.flush()
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data
