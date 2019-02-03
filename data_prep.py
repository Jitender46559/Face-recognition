import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import random

Classification = []
Train_DIR = "Images"
for entry_name in tqdm(os.listdir(Train_DIR)):
    entry_path = os.path.join(Train_DIR, entry_name)
    if os.path.isdir(entry_path):
        Classification.append(entry_name)


Genrelization = []
Test_DIR = "Validation_data"
for entry_name in tqdm(os.listdir(Test_DIR)):
    entry_path = os.path.join(Test_DIR, entry_name)
    if os.path.isdir(entry_path):
        Genrelization.append(entry_name)

Training_data = []


def create_training_data():
    for category in Classification:
        path = os.path.join(Train_DIR, category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                Training_data.append(np.array(img_array))
            except Exception as e:
                print(e)


Testing_data = []


def create_testing_data():
    for category in Genrelization:
        path = os.path.join(Test_DIR, category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                Testing_data.append(np.array(img_array))
            except Exception as e:
                print(e)


create_training_data()
random.shuffle(Training_data)

create_testing_data()

x_train = []
x_label = [1]
for features in Training_data:
    x_train.append(features)

y_test = []
y_label = [1]
for features in Testing_data:
    y_test.append(features)


print(np.shape(x_train))
print(np.shape(y_test))


##Save DATASET
pickle_out = open("training_img.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("training_label.pickle","wb")
pickle.dump(x_label, pickle_out)
pickle_out.close()

pickle_out = open("test_img.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_out = open("test_label.pickle","wb")
pickle.dump(y_label, pickle_out)
pickle_out.close()

##Use DATASET
##pickle_in = open("training_img.pickle","rb")
##var = pickle.load(pickle_in)
