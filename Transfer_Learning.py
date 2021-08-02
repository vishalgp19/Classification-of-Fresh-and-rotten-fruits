from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)


# Freeze the base model
base_model.trainable = False


# Create inputs with correct shape
inputs = keras.Input(shape=(224,224,3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(units=6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs,outputs)

# Compile model
model.compile(loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# Data Augmentation

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    vertical_flip=False,
    horizontal_flip=True
)

# load training dataset
train_it = datagen.flow_from_directory(directory="data/fruits/train/",
                                       target_size=(224, 224),
                                       color_mode='rgb',
                                       class_mode="categorical",
                                       batch_size=32)
# load validation dataset
valid_it = datagen.flow_from_directory(directory="data/fruits/valid/",
                                       target_size=(224, 224),
                                       color_mode='rgb',
                                       class_mode="categorical",
                                       batch_size=32)

# Train Model
history = model.fit(train_it,
                    validation_data=valid_it,
                    steps_per_epoch=train_it.samples/train_it.batch_size,
                    validation_steps=valid_it.samples/valid_it.batch_size,
                    epochs=50)


# Plot the graphs to examine the trajectory of accuracy and loss

train_acc  = history.history['accuracy']
train_loss = history.history['loss']
val_acc    = history.history['val_accuracy']
val_loss   = history.history['val_loss']
epochs     = range(len(train_acc))

# Accuracy
plt.plot(epochs,train_acc)
plt.plot(epochs,val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train_acc','val_acc'], loc='center right')
plt.figure()

# Loss
plt.plot(epochs,train_loss)
plt.plot(epochs,val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train_loss','val_loss'], loc='upper right')
plt.figure()



# Unfreeze the base model
base_model.trainable = True

# Compile the model with a lower learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Again train the whole model (base model including new layers)
history2 = model.fit(train_it,
                    validation_data=valid_it,
                    steps_per_epoch=train_it.samples/train_it.batch_size,
                    validation_steps=valid_it.samples/valid_it.batch_size,
                    epochs=30)

# Plot the graphs to examine the trajectory of accuracy and loss

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(len(train_acc))

# Accuracy
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train_acc', 'val_acc'], loc='center right')
plt.figure()

# Loss
plt.plot(epochs, train_loss)
plt.plot(epochs, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.figure()
