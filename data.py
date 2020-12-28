from tensorflow.keras.datasets import imdb
import numpy as np


class IMDBData:
    def __init__(self, num_words=10000):
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
            num_words=num_words
        )
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_word_index = dict(
            [(value, key) for (key, value) in imdb.get_word_index().items()])

    def decode_review(self, encoded_review):
        # The indices are offset by 3 because 0, 1, and 2 are reserved indices for
        # padding, start of sequence, and unknown
        return ' '.join([self.reverse_word_index.get(i - 3, '?') for i in encoded_review])

    def get_vectorized_train_data_and_labels(self):
        vectorized_train_data = self.__vectorize_sequences(self.train_data)
        labels = np.asarray(self.train_labels).astype('float32')
        return (vectorized_train_data, labels)

    def get_vectorized_test_data_and_labels(self):
        vectorized_test_data = self.__vectorize_sequences(self.test_data)
        labels = np.asarray(self.test_labels).astype('float32')
        return (vectorized_test_data, labels)

    def __vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results
