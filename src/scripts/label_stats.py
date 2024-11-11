import os
import glob

files = glob.glob("/home/yeleussinova/data_1TB/wagons/recognizer/labels/*.txt")

labels = {}
chars = {}
for file in files:
    with open(file, 'r') as f:
        label = f.read()
        if label not in labels.keys():
            labels[label] = 1
        else:
            labels[label] += 1
        for char in label:
            if char not in chars.keys():
                chars[char] = 1
            else:
                chars[char] += 1


sorted_labels = {k: v for k, v in sorted(labels.items(), key=lambda item: item[1])}
sorted_chars = {k: v for k, v in sorted(chars.items(), key=lambda item: item[1])}

print(len(sorted_labels))
print(sorted_labels)
print(len(sorted_chars))
print(sorted_chars)