import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers,models,regularizers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfd
data = {
    "Model": ["ViT", "ResNet", "VGG16"],
    "Number of Epochs": ["100 to 300", "200 to 300", "100 to 200"],
    "Key Techniques": ["Not used", "Used", "Used in newer versions"],
    "Learning Rate": ["Data Augmentation, AdamW Optimizer", "Skip Connenctions Stochastic Depth, Data Agme...", "Dropout,MaxPooling, Data Augmentation"],
    "Patch Size": ["16*16 or 32*32", "-", "-"],
    "Number of Layers(Depth)": ["12 to 24", "18 to 110", "16 or 19"],
    "Hidden Size": ["768 or 1024", "-", "-"],
    "Attention Heads": ["12 to 16", "-", "-"],
    "Batch Size": ["-", "32 or 64", "32 to 128"],
    "Learning Rate": ["0.0001 to 0.001", "0.1(with decay)", "0.01(with decay)"],
    "Weight Decay": ["-", "0.0001", "-"],
    "Dropout": ["-", "-", "0.5 to 0.7"],
    "Accuracy (%)": ["96% to 98%", "96% to 97.5%", "95% to 96%"]
}
df=pd.DataFrame(data)
df
ds, info=tfd.load('svhn_cropped',split=['train','test'],as_supervised=True,with_info=True)
train_ds,test_ds = ds
x_train=np.array([image.numpy() for image, label in train_ds])
y_train=np.array([label.numpy() for image, label in train_ds])
x_test=np.array([image.numpy() for image, label in test_ds])
y_test=np.array([label.numpy() for image, label in test_ds])
x_train,x_test=x_train/255.0,x_test/255.0
datagen=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
datagen.fit(x_train)
class_names=['0','1','2','3','4','5','6','7','8','9']
#VGG16
def creat_model_vgg16(input_shape=(32,32,3), num_classes=10):
    model=models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
vgg16_model = creat_model_vgg16()
vgg16_model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
loss='categorical_crossentropy', metrics=['accuracy'])
vgg16_history = vgg16_model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=20,
validation_data=(x_test, y_test))
def display_predic(model, model_name):
    num_images=5
    indices = np.random.choice(len(x_test), num_images)
    sample_images = x_test[indices]
    sample_labels = y_test[indices]

    predictions = model.predict(sample_images)
    predictions_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"Model: {model_name}\nPredict: {class_names[predictions_classes[i]]}\nTrue: {class_names[true_classes[i]]}")
        plt.axis('off')
    plt.show()
    vgg16_loss,vgg16_accuracy = vgg16_model.evaluate(x_test, y_test)

print(f'vgg16_loss:{vgg16_loss},vgg16_accuracy:{vgg16_accuracy}')
display_predic(vgg16_model, "VGG16")
base_model =tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers[-10:]:
    layer.trainable = True

def create_model_resnet50(input_shape=(32,32,3),num_classes=10):
  model = models.Sequential()
  model.add(base_model)

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(512,activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model
resnet_model=create_model_resnet50()
resnet_model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
loss='categorical_crossentropy', metrics=['accuracy'])
AZxhistory_resnet=resnet_model.fit(datagen.flow(x_train, y_train, batch_size=64),
epochs=20, validation_data=(x_test, y_test))
loss, accuracy = resnet_model.evaluate(x_test, y_test)
print(f'ResNet50 Model - Loss: {loss}, Accuracy: {accuracy}')

display_predic(resnet_model, "ResNet50_pretrain")
display_predic(resnet_model, "ResNet50")
