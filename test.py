import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import joblib
classnames = ['no', 'yes']
dictlabel = {0: 'no', 1: 'yes'}

load_model=joblib.load("final_model.sav")
#model = tf.saved_model.load("/Users/lqiu002/Desktop/project/savedmodel")

print(type(load_model))
def Get_Data_List(path):
    filedir = path + 'p128/'  # 会不会多了一个/
    os.listdir(filedir)
    filelist = []
    for root, dirs, files in os.walk(filedir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filelist.append(os.path.join(root, file))
    return filelist


def plot_image(i, predictions_array, true_label, img):
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

filedir_sample = "/Users/lqiu002/Desktop/project/UI/"
#os.listdir(filedir)
filelist_pred_sample = Get_Data_List(filedir_sample)
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
print(pred_images[0])
predictions= load_model.predict(pred_images)

pred_labels=[0]*80

for i in range(len(filelist_pred_sample)):
    img=pred_images[i]
    plt.imshow(img, cmap=plt.cm.binary)
    img=(np.expand_dims(img,0))
    predictions_single=load_model.predict(img)
    plot_value_array(0,predictions_single,pred_labels)
    _=plt.xticks(range(2),classnames,rotation=45)
    print('第'+str(i+1)+'张图像识别为'+dictlabel[np.argmax(predictions_single[0])])

num_row = 5
num_col = 5
num_image = num_row * num_col
plt.figure(figsize=(2 * 2 * num_col, 2 * num_row))
for i in range(num_image):
    plt.subplot(num_row, 2 * num_col, 2 * i + 1)
    plot_image(i, predictions, pred_labels, pred_images)
    plt.subplot(num_row, 2 * num_col, 2 * i + 2)
    plot_value_array(i, predictions, pred_labels)

plt.show()
