from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class Model:

    def __init__(self, inner_num_units=8, inner_activation='relu'):
        self.model = models.Sequential([
            layers.Dense(inner_num_units, activation=inner_activation),
            layers.Dense(inner_num_units, activation=inner_activation),
            layers.Dense(inner_num_units, activation=inner_activation),
            layers.Dense(1, activation='sigmoid'),
        ])
        self.history = None

    def fit(self, train_data, train_labels, val_data, val_labels, epochs, batch_size=512):
        if not self.model._is_compiled:
            raise Exception('Model is not compiled yet')
        self.history = self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, val_labels),
        )

    def compile(self, optimizer='rmsprop', loss='mse'):
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=['accuracy'])

    def predict(self, test_data):
        if self.history == None:
            raise Exception("Model has not been fitted yet")
        return self.model.predict(test_data)

    def evaluate(self, test_data, test_labels):
        if self.history == None:
            raise Exception("Model has not been fitted yet")
        return self.model.evaluate(test_data, test_labels)

    def plt_train_val_loss(self):
        plt.clf()  # clear the plot
        if self.history == None:
            raise Exception("Model has not been fitted yet")
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        # 'bo' is for blue dot
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        # 'b' is for solid line
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plt_train_val_accuracy(self):
        plt.clf()  # clear the plot
        if self.history == None:
            raise Exception("Model has not been fitted yet")
        history_dict = self.history.history
        print(history_dict.keys())
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
