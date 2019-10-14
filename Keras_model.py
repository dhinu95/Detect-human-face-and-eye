from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

def create_model():
    """
    Inputs:
        None
    Outputs:
        model: compiled keras model
    """
    def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
        regularization = l2(l2_regularization)

        # base
        img_input = Input(input_shape)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

 # module 1
        residual = Conv2D(16, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

 
 # module 2
        residual = Conv2D(32, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])


# module 3
        residual = Conv2D(64, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])


# module 4
        residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        x = Conv2D(num_classes, (3, 3),
                   # kernel_regularizer=regularization,
                   padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        model = Model(img_input, output)
        return model
    model = mini_XCEPTION((48, 48, 1), 6)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
model = create_model()



#Accuracy
def accuracy(x_test, y_test, model):
    """
    Inputs:
        x_test: test samples
        y_test : test labels
        model: keras model
    Ouputs:
        acc: float, accuracy of test data on model
    """
    # YOUR CODE HERE
    acc = model.evaluate(x_test, y_test)[1]
    return acc

acc = accuracy(X_test, y_test, model)
print('Test accuracy is, ', acc*100, '%')
