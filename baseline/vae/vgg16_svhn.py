import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
import numpy as np
from keras import regularizers
from keras.layers.core import Lambda
from keras import backend as K
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import scipy.io
from sklearn.preprocessing import LabelBinarizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels

def svhn_preprocessing(train_images, test_images):
    images = np.concatenate((train_images, test_images), axis = 0)
    mean, std = get_mean_std(images)

    for i in range(train_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images

class svhnvgg:
    def __init__(self, train=False):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('model_weights/vgg16_svhn_weights.h5')
            learning_rate = 0.1
            lr_decay = 1e-6
            sgd = SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            
    def build_model(self):
        model = Sequential()
        weight_decay = self.weight_decay
        model.add(InputLayer(input_shape=self.x_shape))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model
    
    def cifar_preprocessing(self, x_test):
        temp = np.copy(x_test)
        temp = temp.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
        return temp

    def normalize(self, X_train, X_test):
        mean = np.mean(X_train, axis=(0,1,2,3))
        std = np.std(X_train, axis=(0,1,2,3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test
    
    def normalize_production(self, x):
        mean = 120.707
        std = 64.15
        return (x-mean) / (std + 1e-7)
    
    def predict(self, x, normalize=True, batch_size=50):
        if normalize:
            x = self.cifar_preprocessing(x)
        return self.model.predict(x, batch_size)
    
    def train(self, model):
        batch_size = 64
        maxepoches = 200
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # # x_train = x_train.astype('float32')
        # # x_test = x_test.astype('float32')
        # x_train = self.cifar_preprocessing(x_train)
        # x_test = self.cifar_preprocessing(x_test)

        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)
        

        x_train = scipy.io.loadmat('../../dataset/train_32x32.mat')['X'] # 73257
        y_train = scipy.io.loadmat('../../dataset/train_32x32.mat')['y']

        x_test = scipy.io.loadmat('../../dataset/test_32x32.mat')['X'] # 26032 
        y_test = scipy.io.loadmat('../../dataset/test_32x32.mat')['y']

        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.

        x_train, x_test = svhn_preprocessing(x_train, x_test)
        
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_test = lb.fit_transform(y_test)

        
        def lr_schedule(epoch):
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            # print('Learning rate: ', lr)
            return lr

        
        
        # lr_scheduler = LearningRateScheduler(lr_schedule)

        # # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        # #                        cooldown=0,
        # #                        patience=5,
        # #                        min_lr=0.5e-6)

        # lr_reducer = ReduceLROnPlateau(factor=0.2,
        #                        monitor='val_loss',
        #                        patience=2,
        #                        min_lr=0.001)
        def lr_scheduler(epoch):
            return learning_rate * (0.5**(epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        sgd = SGD(learning_rate=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        # optimizer = Adam(lr=lr_schedule(0))
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        save_dir = os.path.join(os.getcwd(), 'saved_vgg16_models')
        model_name = 'svhn_vgg_model.{epoch:03d}.h5' 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        filepath = os.path.join(save_dir, model_name)
        check_point = ModelCheckpoint(filepath=filepath,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True)

        
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            shear_range=0.,  
            zoom_range=0., 
            channel_shift_range=0.,
            fill_mode='nearest', 
            cval=0., 
            rescale=None, 
            preprocessing_function=None, 
            data_format=None, 
            validation_split=0.0
        )
        # testgen = ImageDataGenerator()
        # vali = testgen.flow(x_test, y_test, batch_size=batch_size)
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[check_point, reduce_lr],verbose=1)
        model.save_weights('vgg16_svhn.h5')
        return model
    
if __name__ =='__main__':
    # x_train = scipy.io.loadmat('../../dataset/train_32x32.mat')['X'] # 73257
    # y_train = scipy.io.loadmat('../../dataset/train_32x32.mat')['y']

    # x_test = scipy.io.loadmat('../../dataset/test_32x32.mat')['X'] # 26032 
    # y_test = scipy.io.loadmat('../../dataset/test_32x32.mat')['y']

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255.
    # x_test /= 255.

    # x_train_shape = x_train.shape
    # x_test_shape = x_test.shape

    # x_train = x_train.reshape(x_train_shape[-1], x_train_shape[0], x_train_shape[1], x_train_shape[2])
    # x_test= x_test.reshape(x_test_shape[-1], x_test_shape[0], x_test_shape[1], x_test_shape[2])
    
    # x_train, x_test = svhn_preprocessing(x_train, x_test)

    # y_test = keras.utils.to_categorical(y_test)

    
    model = svhnvgg()

    # predicted_x = model.predict(x_test)
    # residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    # loss = sum(residuals)/len(residuals)
    # print("the validation 0/1 loss is: ",loss)
