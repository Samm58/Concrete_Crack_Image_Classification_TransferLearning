#%%
# -- 1. SETUP --
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications, models
import numpy as np
import matplotlib.pyplot as plt
import os, datetime

#%%
# -- 2. DATA LOADING --
PATH = os.path.join(os.getcwd(), 'dataset')

BATCH_SIZE = 128
IMG_SIZE = (160, 160)

# Load image dataset using keras
train_dataset = tf.keras.utils.image_dataset_from_directory(PATH,
                                                            shuffle=True,
                                                            subset='training',
                                                            validation_split=0.3,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            seed=42)

validation_dataset = tf.keras.utils.image_dataset_from_directory(PATH,
                                                            shuffle=True,
                                                            subset='validation',
                                                            validation_split=0.3,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            seed=42)

#%%
# -- 3. DATA VISUALIZATION --

class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#%%
# -- 4. VALIDATION-TEST SPLITS --

# Further split the validation dataset into validation-test splits
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

#%%
# -- 5. CONVERT DATASET TYPE --

# Convert the tensorflow datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
# -- 6. DATA AUGMENTATION --

# Create a sequential model to perform data augmentation on the fly
# data_augmentation = keras.Sequential([
#   layers.RandomFlip('horizontal'),
#   layers.RandomRotation(0.2),
# ])

# # See the result
# for image, _ in train_dataset.take(1):
#   plt.figure(figsize=(10, 10))
#   first_image = image[0]
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#     plt.imshow(augmented_image[0] / 255)
#     plt.axis('off')

#%%
# -- 7. DATA NORMALIZATION --

# Define a layer to perform data normalization
preprocess_input = applications.mobilenet_v2.preprocess_input

#%%
# -- 8. TRANSFER LEARNING --

# Perform transfer learning
# 1. Load the pretrained model as feature extractor

IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.summary()
keras.utils.plot_model(base_model)

# 2. Set the base_model to become non-trainable
base_model.trainable = False
base_model.summary()


# -- 9. CLASSIFICATION HEAD --

# Define the classification layers
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names), activation='softmax')

#%%
# -- 10. PIPELINE --

# Build the entire model pipeline
# 1. Input
inputs = keras.Input(shape=IMG_SHAPE)

# 2. Data Augmentation
# x = data_augmentation(inputs)

# 3. Data Normalization layer
x = preprocess_input(inputs)

# 4. Transfer learning feature extractor
x = base_model(x, training=False)

# 5. Classification layers
x = global_avg(x)
x = layers.Dropout(0.2)(x)
outputs = output_layer(x)

# 6. Define the full model out
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
# -- 11. MODEL COMPILATION --

# Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Prepare the callback functions for the model training
early_stopping = callbacks.EarlyStopping(patience=5)
logpath = os.path.join('tensorboard_log', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(logpath)
#%%
# -- 12. EVALUATION BEFORE TRAINING --
model.evaluate(validation_dataset)

#%%
# -- 13. MODEL TRAINING --

EPOCH = 10
history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs=EPOCH, callbacks=[early_stopping, tb]
)

#%%
# -- 14. FINE TUNING --

# Model fine tuning by training the top layers of the base model along with the classifier
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

base_model.summary()

#%%
# -- 15. FINE TUNE MODEL COMPILATION --

# Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()

#%%
# -- 16. FINE TUNE MODEL TRAINING --

# Perform the fine tune training
fine_tune_epochs = 10
total_epochs =  EPOCH + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         callbacks=[early_stopping, tb])

#%%
# -- EXTRA: PLOTTING GRAPH --

# Plot the training graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.99, 1])
plt.plot([EPOCH-1,EPOCH-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 0.03])
plt.plot([EPOCH-1,EPOCH-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#%%
# -- 17. FINE TUNE MODEL EVALUATION --
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
# -- 18. MODEL DEPLOYMENT --
# Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)

# Identify the class for the predictions
prediction_indexes = np.argmax(predictions, axis=1)

# Create a label map to map index to class names
label_map = {i:names for i, names in enumerate(class_names)}
prediction_list = [label_map[i] for i in prediction_indexes]
label_list = [label_map[i] for i in label_batch]

# Display the images with the predictions and labels
plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(f'Prediction: {prediction_list[i]}, Label: {label_list[i]}')
  plt.axis("off")
  plt.grid('off')
  plt.tight_layout()

#%%
# -- 19. MODEL SAVING --
  
model.save(os.path.join('models', 'classify.h5'))

#%%

