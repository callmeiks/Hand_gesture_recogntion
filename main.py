import os
from PIL import Image
import tensorflow as tf
from keras.layers import Conv2D
from keras.models import Sequential
from tensorflow import keras
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten
import joblib
import matplotlib.pyplot as plt

def ChangeTo128(filepath):
    filedir = filepath
    os.listdir(filedir)

    filelist1 = []
    for root, dirs, files in os.walk(filedir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filelist1.append(os.path.join(root, file))

    for filename in filelist1:
        try:
            im = Image.open(filename)
            new_im = im.resize((128, 128))
            new_im.save(filepath + '/p128/' + str(filename[38:-4]) + '.jpg')
            print('succeeded')
        except OSError as e:
            print(e.args)


def Get_Data_List(path):
    filedir = path + 'p128/'  # 会不会多了一个/
    os.listdir(filedir)
    filelist = []
    for root, dirs, files in os.walk(filedir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filelist.append(os.path.join(root, file))
    return filelist

    '''这里应该可以直接让视频转图片的那个程序输出128格式的图片，
    而不是在这里处理。
    这里只用作深度学习会更好。
    --------------------------------------------------------------------------------'''



Path_of_Wrong_Picture = "/Users/lqiu002/Desktop/project/picture/"
Path_of_Correct_Picture = "/Users/lqiu002/Desktop/project/gesture/"
Path_of_Predicted_Picture = ""

'''ChangeTo128(Path_of_Correct_Picture)
ChangeTo128(Path_of_Wrong_Picture)'''


filelist1 = Get_Data_List(Path_of_Wrong_Picture)
filelist2 = Get_Data_List(Path_of_Correct_Picture)
listall = filelist1 + filelist2

M = []
for filename in listall:
    im = Image.open(filename)
    width, height = im.size
    im_L = im.convert("L")
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float32') / 255.0
    arr1.shape
    list_img = arr1.tolist()
    M.extend(list_img)

X = np.array(M).reshape((len(listall), width, height))
X.shape

classnames = ['no', '右手']
dictlabel = {0: 'no', 1: '右手'}

label = [0] * len(filelist1) + [1] * len(filelist2)
y = np.array(label)

train_images, test_images, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2, random_state=0)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(classnames[train_labels[i]])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 128)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
#please
model.fit(train_images, train_labels, batch_size=64, epochs=5)

filename='final_model.sav'
joblib.dump(model,filename)
#tf.saved_model.save(model, "/Users/lqiu002/Desktop/project/savedmodel")
'''test_loss, test_acc = model.evaluate(test_images, test_labels)'''
'''print('Test accuracy', test_acc)

predictions = model.predict(test_images)
np.argmax(predictions[0])
dictlabel[np.argmax(predictions[0])]

'''
'''def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = '#00bc57'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classnames[predicted_label],
                                         100 * np.max(predictions_array),
                                         classnames[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(classnames)), predictions_array,
                       color='#FF7F0E', width=0.2)
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("#00bc57")
'''

'''
def test_single_picture(number):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(number, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(number, predictions, test_labels)
    plt.show()
'''




'''ChangeTo128('/Users/lqiu002/Desktop/project/UI/')'''
'''filedir_sample = "D:/MachineLearning/unknown/"'''
#os.listdir(filedir)
'''filelist_pred_sample = Get_Data_List(filedir_sample)
N=[]
for filename in filelist_pred_sample:
    im=Image.open(filename)
    width,height=im.size
    im_L=im.convert("L")
    Core=im_L.getdata()
    arr1=np.array(Core,dtype='float')/255.0
    arr1.shape
    list_img=arr1.tolist()
    N.extend(list_img)

pred_images=np.array(N).reshape(len(filelist_pred_sample),width,height)
predictions= model.predict(pred_images)

pred_labels=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(len(filelist_pred_sample)):
    img=pred_images[i]
    plt.imshow(img, cmap=plt.cm.binary)
    img=(np.expand_dims(img,0))
    predictions_single=model.predict(img)
    plot_value_array(0,predictions_single,pred_labels)
    _=plt.xticks(range(2),classnames,rotation=45)
    print('第'+str(i+1)+'张图像识别为'+dictlabel[np.argmax(predictions_single[0])])
'''



'''for root, dirs, files in os.walk(filedir):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            filelist_pred.append(os.path.join(root, file))'''

'''num_row = 5
num_col = 3
num_image = num_row * num_col
plt.figure(figsize=(2 * 2 * num_col, 2 * num_row))
for i in range(num_image):
    plt.subplot(num_row, 2 * num_col, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_row, 2 * num_col, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

plt.show()
'''