import data_manager
import copy


data = data_manager.load_data("/home/igor/PycharmProjects/itComesHandy/data")
y = data_manager.add_category(data)

train_x, train_y, test_x, test_y = data_manager.split_data(data, y)

train_x = data_manager.normalize_data(train_x)
test_x = data_manager.normalize_data(test_x)


