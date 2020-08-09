from read_dataset import read_data, read_labels, convert_to_ascii
import os
import numpy as np

path_list = ["/Train/"]  # ,"/Test/","/Validate/"]
label_list = ["Train_labels.txt"]  # ,"Test_labels.txt","Validate_labels.txt"]

for path in path_list:
    x = read_data(os.getcwd() + path)
    x = np.asarray([i[1].T for i in sorted(x.items())])
for label in label_list:
    y = read_labels(label)
    y = np.array([np.array(convert_to_ascii(i)) for i in y.values()])

np.save("labels.npy", y)

np.save("train_data.npy", x)
