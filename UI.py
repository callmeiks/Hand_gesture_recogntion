import PIL
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk


root=Tk()
root.title("Hand Sign Reader")
root.geometry("1000x1000")
root.configure(bg="white")
Label(root, text="Hand sign reader", font=("times new roman",30,"bold"),bg="black", fg="white").pack()

f1=LabelFrame(root, bg="red")
f1.pack()
L1=Label(f1,bg="red")
L1.pack()


cap=cv2.VideoCapture(1)
b = Button(root, text="stop Video",command=lambda: root.quit())
b.place(relx=0.35, rely=0.8, relwidth=0.3, relheight=0.1)
b = b.pack(fill=X,expand=True)
count = 0
time_interval=3
frames_save_path="/Users/lqiu002/Desktop/project/UI"

imgnum=0

while count<110:

    success,img=cap.read()

    img2= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2=cv2.flip(img2,1)
    img2=PIL.ImageTk.PhotoImage(Image.fromarray(img2))
    L1['image']=img2
    root.update()
    count += 1
    #print(count)
    if count % time_interval==0:
        imgnum +=1
        file_path=frames_save_path + "/frame_"+str(imgnum)+".jpg"
        cv2.imwrite(file_path,img)
        #print("Image "+str(imgnum)+" succesfully converted to file!")

print("exit")
cap.release()

