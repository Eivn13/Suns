import pickle
import numpy as np


train_data = pickle.load(open('gdrive/My Drive/dataset/train50x50.pickle', 'rb'))
valid_data = pickle.load(open('gdrive/My Drive/dataset/valid50x50.pickle', 'rb'))
test_data = pickle.load(open('gdrive/My Drive/dataset/test50x50.pickle', 'rb'))

train_img = []
train_labels = []
valid_img = []
valid_labels = []
test_img = []
test_labels = []

# tato funkcia pripravi dataset do numpy arr v shape (7500,)
global train_data, valid_data, test_data
global train_img, train_labels, valid_img, valid_labels, test_img, test_labels
tmp_data = []
tmp_labels = []
for tuple in train_data:
  img = tuple[0].reshape(50*50*3)
  tmp_data.append(img)
  tmp_labels.append(tuple[1])
global train_img, train_labels
train_img = np.array(tmp_data)
train_labels = np.array(tmp_labels)

tmp_data = []
tmp_labels = []
for tuple in valid_data:
  img = tuple[0].reshape(50*50*3)
  tmp_data.append(img)
  tmp_labels.append(tuple[1])
global valid_img, valid_labels
valid_img = np.array(tmp_data)
valid_labels = np.array(tmp_labels)

tmp_data = []
tmp_labels = []
for tuple in test_data:
  img = tuple[0].reshape(50*50*3)
  tmp_data.append(img)
  tmp_labels.append(tuple[1])
global test_img, test_labels
test_img = np.array(tmp_data)
test_labels = np.array(tmp_labels)

pickle.dump(train_img, open("gdrive/My Drive/dataset/train_img.p", "wb"))
pickle.dump(train_labels, open("gdrive/My Drive/dataset/train_labels.p", "wb"))
pickle.dump(valid_img, open("gdrive/My Drive/dataset/valid_img.p", "wb"))
pickle.dump(valid_labels, open("gdrive/My Drive/dataset/valid_labels", "wb"))
pickle.dump(test_img, open("gdrive/My Drive/dataset/test_img.p", "wb"))
pickle.dump(test_labels, open("gdrive/My Drive/dataset/test_labels.p", "wb"))
