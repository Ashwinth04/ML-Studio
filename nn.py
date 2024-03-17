import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,MobileNetV2,ResNet50
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import os
import random
import shutil

class NeuralNetwork:

    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.training_dir = os.path.join('directory','training')
        self.validation_dir = os.path.join('directory','validation')
        self.history = None
        self.train_gen = None
        self.val_gen = None

    def make_train_val_dirs(self):
        os.makedirs('directory',exist_ok=True)
        os.makedirs(os.path.join('directory','training'),exist_ok=True)
        os.makedirs(os.path.join('directory','validation'),exist_ok=True)
        classes = os.listdir(self.root_dir)
        num_classes = len(classes)
        for class_name in classes:
            os.makedirs(os.path.join('directory','training',class_name),exist_ok=True)
            os.makedirs(os.path.join('directory','validation',class_name),exist_ok=True)
        print("Training and validation directories created!")

    def create_dataset(self,split_size = 0.8):
        for dir_name in os.listdir(self.root_dir):
            samples = os.listdir(os.path.join(self.root_dir,dir_name))
            random.shuffle(samples)
            split_no = int(len(samples)*split_size)
            train_samples = samples[:split_no]
            val_samples = samples[split_no:]
            for sample in train_samples:
                s = os.path.join(self.root_dir,dir_name,sample)
                d = os.path.join('directory','training',dir_name,sample)
                shutil.copyfile(s,d)
                
            for sample in val_samples:
                s = os.path.join(self.root_dir,dir_name,sample)
                d = os.path.join('directory','validation',dir_name,sample)
                shutil.copyfile(s,d)

        print("Dataset created!")

    def train_val_gens(self):
        mode = "categorical"
        
        train_datagen = ImageDataGenerator(rescale=1./255.,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
        
        train_generator = train_datagen.flow_from_directory(self.training_dir,
                                                            batch_size=20,
                                                            class_mode=mode,
                                                            target_size=(150,150))
        
        validation_datagen = ImageDataGenerator(rescale=1./255.)
                        
        
        validation_generator = validation_datagen.flow_from_directory(self.validation_dir,
                                                                    batch_size=20,
                                                                    class_mode=mode,
                                                                    target_size=(150,150))
        
        self.train_gen = train_generator
        self.val_gen = validation_generator

        print("Generators created !")
    

    def model_vgg(self):
        num_classes = len(os.listdir(os.path.join('directory', 'training')))
        base_model = VGG16(weights = 'imagenet',include_top = False,input_shape = (150,150,3))
        for layer in base_model.layers:
            layer.trainable = False
            
        X = tensorflow.keras.layers.Flatten()(base_model.output)
        X = tensorflow.keras.layers.Dense(256,activation = 'relu')(X)
        output = tensorflow.keras.layers.Dense(num_classes,activation = 'softmax')(X)
        model = Model(inputs = base_model.input,outputs = output)
        model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy','precision','recall'])
        history = model.fit(self.train_gen,validation_data = self.val_gen,epochs = 10)
        self.history = history
        model.save('vgg_model.h5')
        print("Model is trained")

    def model_mobilenetv2(self):
        num_classes = len(os.listdir(os.path.join('directory', 'training')))
        base_model = MobileNetV2(weights = 'imagenet',include_top = False,input_shape = (150,150,3))
        for layer in base_model.layers:
            layer.trainable = False
            
        X = base_model.output
        X = tensorflow.keras.layers.GlobalAveragePooling2D()(X)
        X = tensorflow.keras.layers.Dense(1024,activation='relu')(X)
        output = tensorflow.keras.layers.Dense(num_classes,activation = 'softmax')(X)
        model = Model(inputs = base_model.input,outputs = output)
        model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy','precision','recall'])
        history = model.fit(self.train_gen,validation_data = self.val_gen,epochs = 10)
        self.history = history
        model.save('mobilenet_model.h5')
        print("Model is trained")

    def resnet(self):
        num_classes = len(os.listdir(os.path.join('directory', 'training')))
        base_model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (150,150,3))
        for layer in base_model.layers:
            layer.trainable = False
            
        X = base_model.output
        X = tensorflow.keras.layers.GlobalAveragePooling2D()(X)
        X = tensorflow.keras.layers.Dense(1024,activation='relu')(X)
        output = tensorflow.keras.layers.Dense(num_classes,activation = 'softmax')(X)
        model = Model(inputs = base_model.input,outputs = output)
        model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy','precision','recall'])
        history = model.fit(self.train_gen,validation_data = self.val_gen,epochs = 10)
        self.history = history
        model.save('resnet_model.h5')
        print("Model is trained")
    # def predict(self,path_to_img):
    #         model = load_model('classification_model.h5')
    #         img = cv2.imread(path_to_img)
    #         img = img / 255.0
    #         resized_img = tensorflow.image.resize(img, (150, 150))
    #         yhat = model.predict(np.expand_dims(resized_img, 0))
    #         index = np.argmax(yhat)
    #         class_list = os.listdir(os.path.join('directory','training'))
    #         return class_list[index]
    
# m = NeuralNetwork(os.path.join('image_classification','PetImages'))
# m.make_train_val_dirs()
# m.create_dataset()
# m.train_val_gens()
# m.model()
