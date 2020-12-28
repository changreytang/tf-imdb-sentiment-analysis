from data import IMDBData

imdb_data = IMDBData()

(train_data, train_labels) = imdb_data.get_vectorized_test_data_and_labels()


