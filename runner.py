from data import IMDBData
from model import Model

imdb_data = IMDBData()
model = Model()

(train_data, train_labels) = imdb_data.get_vectorized_train_data_and_labels()
val_data = train_data[:10000]
val_labels = train_labels[:10000]
train_data = train_data[10000:]
train_labels = train_labels[10000:]

model.compile()
model.fit(train_data, train_labels, val_data, val_labels, epochs=4)

model.plt_train_val_loss()
model.plt_train_val_accuracy()

(test_data, test_labels) = imdb_data.get_vectorized_test_data_and_labels()
print(model.evaluate(test_data, test_labels)) # test loss, test accuracy
