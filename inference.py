import os
import json

fake_dir = ''
labels_file = 'labels.json'
labels = {}
label = '1'

for filename in os.listdir(fake_dir):
    file_name = filename.split('.')[0]
    labels[filename] = label
    