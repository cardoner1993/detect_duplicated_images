import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imutils import paths
from pathlib import Path
import pickle
import datetime

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, model_from_yaml
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def image_to_array(folder_path, result, image_max=None):
    """
    input: folder path
    output: X and y matrices
    X - image matrix;
    y - labels/result: 0 as clear and 1 as blur
    image_max - max number of images to include
    """
    X, y = [], []

    for index, file_name in enumerate(list(paths.list_images(folder_path))):
        img = image.load_img(file_name, target_size=(200, 200))
        X.append(np.asarray(img))
        y.append(result)  # 0 for clear; 1 for blurry

    print('---\n', len(X), 'Images in this folder\n')

    return X, y


def normalize_data(X, y):
    X = np.stack(X)
    X = X.astype('float32')
    X = X / 255

    y = to_categorical(y)

    print(f"X shape {np.array(X).shape}. Y shape {np.array(y).shape}")

    return X, y


def split_data(X, y, size=0.8):

    X_shuffled, y_shuffled = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, train_size=int(size * len(X)), random_state=42)

    print('Dimensions\n---')
    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)

    return X_train, X_test, y_train, y_test


def cnn_model(X_train, y_train, output='models/cnn_model'):

    workdir = Path(output) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(workdir, exist_ok=True)

    input_dimension = (200, 200, 3)  # 200x200 RGB image (200, 200)

    es = EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # patience =number of epochs with no improvement after which training will be stopped
        verbose=1,
        mode='auto'
    )
    mcheck = ModelCheckpoint(
        workdir / 'model.h5',  # string, path to save the model file,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1
    )

    model = Sequential()

    # Layer 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_dimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # to avoid overfitting

    # Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    # Layer 3
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 4
    model.add(Dense(2, activation='softmax'))

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=20, epochs=50,  verbose=1,
              validation_split=0.2, callbacks=[mcheck, es])

    return model


def save(model, path):

    output_dir = Path(path)
    if not output_dir.exists():
        Path.mkdir(output_dir, parents=True, exist_ok=True)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(output_dir / 'model_config.yaml', 'w') as file:
        file.write(model_yaml)


def load(path):

    output_dir = Path(path)
    with open(output_dir / 'model_config.yaml') as file:
        model_config = file.read()

    model = model_from_yaml(model_config)
    model.load_weights(output_dir / "model.h5")

    return model


if __name__ == '__main__':

    X, Y = [], []

    train_sets = {
        'paths': ['/home/daca/Documents/detect_duplicated_images/data/test_images_non_blurry',
                  '/home/daca/Documents/detect_duplicated_images/data/test_images_blurry'],
        'class': ['0', '1']
    }

    for item in range(len(train_sets['paths'])):
        x, y = image_to_array(train_sets['paths'][item], train_sets['class'][item])
        X.extend(x), Y.extend(y)

    X, Y = normalize_data(X, Y)
    X_train, X_test, y_train, y_test = split_data(X, Y)
    model = cnn_model(X_train, y_train)
    save(model, path='models/cnn_model')

    score = model.evaluate(X_test, y_test, verbose=1)
    print('accuracy: ', score[1])
    print('loss: ', score[0])
