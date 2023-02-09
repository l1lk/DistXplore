from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from keras.layers import Input

img_rows, img_cols = 28, 28
img_dim = img_rows * img_cols
input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)

model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)
print(model1.summary())
print(model2.summary())
print(model3.summary())