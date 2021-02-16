
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import tensorflow as tf
import numpy as np

SEED = 1234
tf.random.set_seed(SEED)  

# Get current working directory
cwd = os.getcwd()

from google.colab import drive
#drive.mount("/content/drive", force_remount=True)
drive.mount('/content/drive')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = False

# Create training ImageDataGenerator object
if apply_data_augmentation:
    train_data_gen = ImageDataGenerator(rotation_range=10,
                                  
                                        width_shift_range=10,
                                        height_shift_range=10,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        rescale=1./255)
else:
    train_data_gen = ImageDataGenerator(rescale=1./255)

# Create validation and test ImageDataGenerator objects
valid_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

!unzip '/content/drive/My Drive/data_trad.zip'

!ls /content

# Create generators to read images from dataset directory
# -------------------------------------------------------
dataset_dir = os.path.join('/content/data_trad')

# Batch size
bs = 8

# img shape
img_h = 28
img_w = 28

num_classes=2

decide_class_indices = False
if decide_class_indices:
    classes = ['0','1']        # 3
else:
    classes=None

# Training
training_dir = dataset_dir+'/training'
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               classes=classes,
                                               target_size=(28,28),
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED)  # targets are directly converted into one-hot vectors
val_dir = dataset_dir+'/validation'
valid_gen = valid_data_gen.flow_from_directory(val_dir,
                                               batch_size=bs,
                                               classes=classes,
                                               target_size=(28,28),
                                               class_mode='categorical',
                                               shuffle=False,
                                               seed=SEED) 

# Test
test_dir = os.path.join(dataset_dir, 'test')
test_gen = test_data_gen.flow_from_directory(test_dir,
                                             batch_size=bs, 
                                             classes=classes,
                                               target_size=(28,28),
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=SEED)



# Check how keras assigned the labels
train_gen.class_indices

# Create Dataset objects
# ----------------------

# Training
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, 2]))

valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, 2]))
test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, 2]))
# Shuffle (Already done in generator..)
# train_dataset = train_dataset.shuffle(buffer_size=len(train_gen))

# Normalize images (Already done in generator..)
# def normalize_img(x_, y_):
#     return tf.cast(x_, tf.float32) / 255., y_

# train_dataset = train_dataset.map(normalize_img)

# 1-hot encoding <- for categorical cross entropy (Already done in generator..)
# def to_categorical(x_, y_):
#     return x_, tf.one_hot(y_, depth=10)

# train_dataset = train_dataset.map(to_categorical)

# Divide in batches (Already done in generator..)
# train_dataset = train_dataset.batch(bs)

# Repeat
# Without calling the repeat function the dataset 
# will be empty after consuming all the images
train_dataset = train_dataset.repeat()
valid_dataset=valid_dataset.repeat()
test_dataset=test_dataset

# Validation
# ----------

train_dataset

# Architecture: Features extraction -> Classifier


model = tf.keras.Sequential()

# Features extraction
img_h=28
img_w=28
input_shape = [img_h, img_w, 3]

28*28*2


model.add(tf.keras.Input(shape=input_shape))
# Classifier
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

# Visualize created model as a table
model.summary()

# Visualize initialized weights

# Optimization params
# -------------------

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# learning rate
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(x=train_dataset,
          epochs=14,  #### set repeat in training dataset
          steps_per_epoch=len(train_gen),
          validation_data=valid_dataset,
          validation_steps=len(valid_gen))
#         callbacks=callbacks)

# How to visualize Tensorboard

# 1. tensorboard --logdir EXPERIMENTS_DIR --port PORT     <- from terminal
# 2. localhost:PORT   <- in your browser

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(range(len(train_acc)),train_acc)
plt.plot(range(len(train_acc)),val_acc)
plt.legend(labels=('train','valid'))
plt.savefig('acc_traditional.png')

# Let's visualize the activations of our network
from PIL import Image

test_iter = iter(test_dataset)

# Get a test image
test_img = next(test_iter)[0]
test_img = test_img[0]

# Visualize the image
Image.fromarray(np.uint8(np.array(test_img)*255.))

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
def display_activation(activations, act_index): 
    # activations: list of all the activations
    # act_index: the layer we want to visualize (an int in [0, network depth) )
    activation = activations[act_index]
    activation = tf.image.resize(activation, size=[128, 128])
    col_size = activations[0].shape[-1]
    row_size = 1 + act_index
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(8*2.5, 8*1.5), squeeze=False)
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

# Print Confusion Matrix and Classification Report (Precision, Recall, and F1-score)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

Y_prediction = model.predict_generator(test_gen, len(test_gen))
# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_prediction,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = test_gen.classes
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
class_report = classification_report(Y_true, Y_pred_classes, 
                                     target_names=test_gen.class_indices.keys())  # target_names must be ordered depending on the class labels

                                     
print('Confusion Matrix:')
print(confusion_mtx)
print()
print('Classification Report:')
print(class_report)

Y_prediction = model.predict_generator(test_gen, len(test_gen))
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction,axis = 1) 

# Convert validation observations to one hot vectors
Y_true = test_gen.classes
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
confusion_mtx

for i in range(len(Y_true)):
  if Y_true[i]==Y_pred_classes[i]:
    print(i)

# Let's visualize the activations of our network
from PIL import Image

test_iter = iter(test_dataset)

reset(test_iter)

# Get a test image
test_img = next(test_iter)[0]
test_i= test_img[0]
# Visualize the image
#Image.fromarray(np.uint8(np.array(test_img)*255.))

# Get a test image
import tensorflow as tf
import torch


te=tf.convert_to_tensor([0,test_i])
pred = model.predict()
Y_pred_class = np.argmax(pred,axis=1)
lab_img, Y_pred_class

#### ROC curve construction
#### We have in prob the probabilities, extracted via softmax, of the two classe, look how sensitivity and specificity changes as we chane the threshold
import matplotlib.pyplot as plt
num_it=300
TP=np.zeros_like(range(num_it+1))
FP=np.zeros_like(range(num_it+1))

for p in range(num_it+1):
  cnt=0
  for im in range(120):
    if (p/num_it)>Y_prediction[im,0]:
      pred=1
    else:
      pred=0
    if pred==Y_true[im]:
      if pred==1:
        TP[p]+=1
    if pred!=Y_true[im]:
      if pred==1:
        FP[p]+=1
        

      
TP=TP/60
FP=FP/60

plt.plot([0,1],[0,1])
plt.plot(FP,TP)
plt.title('ROC curve', fontsize=8)
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')

plt.savefig('ROC_net_trad.png')

import random
import numpy as np
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 80
    deviation = VARIABILITY
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img
# Prepare data-augmenting data generator
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        preprocessing_function=add_noise,
    )

import random
import numpy as np
def add_noise(img):
    '''Add random noise to an image'''
    noise = np.random.uniform(-25,25,img.shape)
 
    img += np.floor(noise)

    img-=np.min(img)
    img=np.floor(img/np.max(img)*255.)
    print((img))
    return img
# Prepare data-augmenting data generator
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    preprocessing_function=add_noise,
        rescale=1./255
        
    )

(80/255)*(80/255)

!unzip '/content/drive/My Drive/100.zip'

# Create Dataset objects
# ----------------------
# Test
test_dir = os.path.join('/content/1000')
test_gen = datagen.flow_from_directory(test_dir,
                                             batch_size=bs, 
                                               target_size=(28,28),
                                             class_mode='categorical',
                                             shuffle=False,
                                             seed=SEED)
test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, 2]))
# Shuffle (Already done in generator..)
# train_dataset = train_dataset.shuffle(buffer_size=len(train_gen))

# Normalize images (Already done in generator..)
# def normalize_img(x_, y_):
#     return tf.cast(x_, tf.float32) / 255., y_

# train_dataset = train_dataset.map(normalize_img)

# 1-hot encoding <- for categorical cross entropy (Already done in generator..)
# def to_categorical(x_, y_):
#     return x_, tf.one_hot(y_, depth=10)

# train_dataset = train_dataset.map(to_categorical)

# Divide in batches (Already done in generator..)
# train_dataset = train_dataset.batch(bs)


test_dataset=test_dataset

Y_prediction = model.predict_generator(test_gen, len(test_gen))
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction,axis = 1) 
sum(Y_pred_classes)/100
np.var(Y_prediction[:,1])
plt.xlim((0,1))
plt.hist(Y_prediction[:,1], bins=20, alpha=0.5)

plt.savefig('hist_trad_noisy')

85/255

