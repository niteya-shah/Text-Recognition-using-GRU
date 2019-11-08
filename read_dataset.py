import os
import skimage.io as io
import numpy as np
import pandas as pd

def read_data(path):
    imgs = {}
    for img in os.listdir(path):
        imgs[int(img.strip(".jpg"))] = io.imread(os.path.join(path, img), as_gray = True)
    return imgs

def read_labels(path):
    imgs = {}
    with open(path) as file:
        for line in file:
            imgs[int(line[0:line.find(" ")].strip(".jpg"))] = line[line.find(" ") + 1:].strip("\n")
    return imgs

def convert_to_ascii(string_vector):
    x = [ord(val) for val in string_vector];
    for i in range(len(x)):
        if x[i] == 32:
            x[i] = 54
        elif (x[i] >= 97 and x[i] <= 122):
            x[i] -= 96
        elif (x[i] >= 65 and x[i] <= 90):
            x[i] -= 38
        elif x[i] == 45:
            x[i] = 53
    return x

def convert_to_char(string_vector):
    string = []
    for i in string_vector:
        if i == 54:
            string.append(" ")
        elif i == 53:
            string.append("-")
        elif (i >= 0 and i <= 25):
            string.append(chr(i + 96))
        elif (i >= 26 and i <= 51):
            string.append(chr(i + 38))
    return string
