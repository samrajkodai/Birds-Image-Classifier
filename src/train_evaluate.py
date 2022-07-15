import imp
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import argparse
from get_data import read_params,get_data


def evaluate(config_path):

    config = read_params(config_path)

    IMAGE_SIZE=[224,224]

    train_path=config['load_data']['train_images']
    test_path=config['load_data']['test_images']
    activation = config['dl_params']['activation']
    optimizer = config['dl_params']['optimizer']
    loss = config['dl_params']['loss']
    metrics = config['dl_params']['metrics']
    epochs = config['dl_params']['epochs']
    batch_size = config['dl_params']['batch_size']

    # add preprocessing layer to the front of VGG
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable=False
        
    folders=glob(f"{train_path}/*")

    print("folders",folders)

    x=Flatten()(vgg.output)

    predictions=Dense(len(folders),activation=activation)(x)

    model = Model(inputs=vgg.input, outputs=predictions)

    model.summary()

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)



    train_datagen=ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)


    training_set=train_datagen.flow_from_directory(directory=train_path,
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical'
                                                )

    test_set=test_datagen.flow_from_directory(directory=test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


    r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=epochs,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
    )




if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="config/params.yaml")
    parsed_args=args.parse_args()
    data=evaluate(config_path=parsed_args.config)