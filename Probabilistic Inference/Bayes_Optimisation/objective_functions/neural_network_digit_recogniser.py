from typing import Tuple, Union

import numpy as np
from keras.utils import np_utils
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

from objective_functions.abstract_objective_function import ObjectiveFunction
from objective_functions.parameter_category import TypeVariable


class NeuralNetworkDigitRecogniser(ObjectiveFunction):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Scale the data to between 0 and 1
    X_train = X_train / 255
    X_test = X_test / 255

    # Flatten arrays from (28x28) to (784x1)
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    # Convert the y's to categorical to use with the softmax classifier
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Establish the input shape for our Networks.
    input_shape = X_train[0].shape

    def evaluate_without_noise(self,
                               data_points: np.ndarray
                               ) -> Union[np.ndarray, float]:
        evaluations = []
        for data_point in data_points:
            log_learning_rate, \
            num_dense_layers, \
            num_input_nodes, \
            num_dense_nodes, \
            batch_size, \
            log_adam_decay = data_point

            learning_rate = np.exp(log_learning_rate)
            num_dense_layers = int(num_dense_layers)
            num_input_nodes = int(num_input_nodes)
            num_dense_nodes = int(num_dense_nodes)
            batch_size = int(batch_size)
            adam_decay = np.exp(log_adam_decay)

            print(log_learning_rate,
                  num_dense_layers,
                  num_input_nodes,
                  num_dense_nodes,
                  batch_size,
                  log_adam_decay)

            activation_function = 'relu'

            evaluations.append(self._evaluate_without_noise(learning_rate,
                                                            num_dense_layers,
                                                            num_input_nodes,
                                                            num_dense_nodes,
                                                            activation_function,
                                                            batch_size,
                                                            adam_decay))

        return np.asarray(evaluations).reshape((-1, 1))

    def _evaluate_without_noise(self,
                                learning_rate,
                                num_dense_layers,
                                num_input_nodes,
                                num_dense_nodes,
                                activation,
                                batch_size,
                                adam_decay
                                ) -> float:
        model = self.create_model(learning_rate, num_dense_layers, num_input_nodes, num_dense_nodes, activation,
                                  adam_decay)
        # named blackbox becuase it represents the structure
        blackbox = model.fit(x=self.X_train,
                             y=self.y_train,
                             epochs=3,
                             batch_size=batch_size,
                             validation_split=0.15,
                             )
        # return the validation accuracy for the last epoch.
        accuracy = blackbox.history['val_accuracy'][-1]

        # Print the classification accuracy.
        print()
        print("Accuracy: {0:.2%}".format(accuracy))
        print()

        # Delete the Keras model with these hyper-parameters from memory.
        del model

        # Clear the Keras session, otherwise it will keep adding new
        # models to the same TensorFlow graph each time we create
        # a model with a different set of hyper-parameters.
        K.clear_session()
        tensorflow.compat.v1.reset_default_graph()

        # the optimizer aims for the lowest score, so we return our negative accuracy
        return -accuracy

    @property
    def dataset_bounds(self) -> Tuple[Tuple[Tuple[float, float],
                                            TypeVariable],
                                      ...]:
        """
        Defines the bounds and the types of variables for the objective function

        Example:
        if dataset_bounds is equal to
        (
        ((1, 2), TypeVariable.REAL),
        ((5, 10), TypeVariable.INTEGER),
        )
        then it means the objective function depends on 2 variables:
        - the first one is a real number between 1 and 2
        - the second one is an integer between 5 (included) and 10 (excluded)
        """

        return (
            ((np.log(1e-4), np.log(1e-2)), TypeVariable.REAL),  # log Learning rate
            ((1, 5 + 1), TypeVariable.INTEGER),  # Number dense layers
            ((1, 512 + 1), TypeVariable.INTEGER),  # Number input nodes
            ((1, 28 + 1), TypeVariable.INTEGER),  # Number dense nodes
            ((1, 128 + 1), TypeVariable.INTEGER),  # batch size
            ((np.log(1e-6), np.log(1e-2)), TypeVariable.REAL)  # log adam decay
        )

    @classmethod
    def create_model(cls,
                     learning_rate,
                     num_dense_layers,
                     num_input_nodes,
                     num_dense_nodes,
                     activation,
                     adam_decay):
        # start the model making process and create our first layer
        model = Sequential()
        model.add(Dense(num_input_nodes, input_shape=cls.input_shape, activation=activation
                        ))
        # create a loop making a new dense layer for the amount passed to this model.
        # naming the layers helps avoid tensorflow error deep in the stack trace.
        for i in range(num_dense_layers):
            name = 'layer_dense_{0}'.format(i + 1)
            model.add(Dense(num_dense_nodes,
                            activation=activation,
                            name=name
                            ))
        # add our classification layer.
        model.add(Dense(10, activation='softmax'))

        # setup our optimizer and compile
        adam = Adam(lr=learning_rate, decay=adam_decay)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
