
from pyexpat import model
from time import sleep
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import pytesseract

from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #busca cuda, sino hay escoge CPU
print(device)


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


print(train_data)
print(train_data.data.size())
print(train_data.targets.size())


plt.imshow(train_data.data[9], cmap='gray')
plt.title('%i' % train_data.targets[0])
#plt.show()

loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, #de las 60000 organiza un batch_size /en bloques de 100 para pasarle en bloques de entrenamientoq
                                          shuffle=True, 
                                          ),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          ),
}
print(loaders)

model_cnn = torch.load("model_cnn.pt")
model_cnn.eval()


sample = next(iter(loaders['test']))
imgs, lbls = sample
actual_number = lbls[:10].numpy()
actual_number
test_output, last_layer = model_cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')


cap = cv2.VideoCapture("araxes1.mp4")
cap.set(1,2200); # Where frame_no is the frame you want

box1_x = int(1920/2) + 180
box1_y = int(1080/2) - 80 #-200 8, -80 7, +40 6, +160 5,
box1_w = 220
box1_h = 130

box2_x = box1_x + 220
box2_y = box1_y
box2_w = 200
box2_h = 100


# #declaro 1 obj tipo trnsforms de 
transform = transforms.ToTensor()
transform_batch = transforms.Compose([
    ToTensor()
])

def flatten(t):
  f = torch.reshape(t, (t.shape[0], 1, -1))
  return f.squeeze()

while True:
    ss, frame = cap.read()

    #frame = cv2.imread("frame.png")

    if ss == True:
        frame_copy = frame.copy()
        frame_box1 = frame_copy[box1_y-int(box1_h/2):box1_y+int(box1_h/2), box1_x-int(box1_w/2):box1_x+int(box1_w/2)]
        frame_box2 = frame_copy[box2_y-int(box2_h/2):box2_y+int(box2_h/2), box2_x-int(box2_w/2):box2_x+int(box2_w/2)]
        gray_frame_box1 = cv2.cvtColor(frame_box1, cv2.COLOR_BGR2GRAY) 
        gray_frame_box2 = cv2.cvtColor(frame_box2, cv2.COLOR_BGR2GRAY) 

        gray_frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY) 
        laplacian = cv2.Laplacian(gray_frame_copy, cv2.CV_16S, ksize=3)
        laplacian = cv2.convertScaleAbs(laplacian)
        cv2.imshow('laplacian',laplacian)

        frame_bin = cv2.threshold(gray_frame_copy, 200, 255, cv2.THRESH_BINARY)[1]

        gauss = cv2.GaussianBlur(frame_bin, (5,5), 0) #El desenfoque gaussiano es muy eficaz para eliminar el ruido gaussiano de una imagen.
        canny = cv2.Canny(gauss, 50, 60) #Detección de bordes 

        (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame,contornos,-1,(0,0,255), 2)
        
        box_cont = 0
        box_array = []

        for c in contornos:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            envoltura = cv2.convexHull(c)
            
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            
            if area > 9000: #10000
                box_cont += 1
                box_array.append(box)                
                cv2.drawContours(frame,[box],0,(0,0,255),10)
                #cv2.drawContours(frame,envoltura,-1,(255,0,255), 8)
        #print("He encontrado {} objetos".format(len(contornos)))
        #sleep(0.5)

        #print(box_cont)
        

        if box_cont >= 1:
            box_arr_tmp = []
                       
            for coo in box_array:
                box_arr_tmp.append(coo[1][1])
            
            pos_box_min = box_arr_tmp.index(max(box_arr_tmp))
            #print(box_array[pos_box_min])
            #frame_detect = frame[box_array[pos_box_min][1][1]+30:box_array[pos_box_min][3][1]-5,box_array[pos_box_min][1][0]+100:box_array[pos_box_min][3][0]-10]

            frame_detect = frame[box_array[pos_box_min][1][1]+30:box_array[pos_box_min][3][1]-10,box_array[pos_box_min][1][0]+100:box_array[pos_box_min][3][0]-9]

            #print(frame_detect.shape[1])
            #M = cv2.getRotationMatrix2D((frame_detect.shape[1]//2,frame_detect.shape[0]//2),3,1)
            #frame_detect = cv2.warpAffine(frame_detect,M,(frame_detect.shape[1],frame_detect.shape[0]))

            #gray_frame_detect = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2GRAY)
            #frame_bin_detect = cv2.threshold(gray_frame_detect, 200, 255, cv2.THRESH_BINARY)[1]

            if frame_detect.shape[1] != 0:
                detect_pytess = pytesseract.image_to_string(frame_detect,config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
                print('pytesseract {}'.format(detect_pytess))


                gray_frame_tensor = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2GRAY)
                gray_frame_tensor = cv2.threshold(gray_frame_tensor, 200, 255, cv2.THRESH_BINARY)[1]
                #gray_frame_tensor = cv2.bitwise_not(gray_frame_tensor)  
                
                cv2.imshow("detect2",gray_frame_tensor)
                frame_box1_resized_detect = cv2.resize(gray_frame_tensor, (28,28), interpolation = cv2.INTER_AREA)
                box1_detect_tensor = transform(frame_box1_resized_detect)
                #print(box1_detect_tensor.shape)
                box1_detect_tensor = torch.reshape(box1_detect_tensor, (1,1,28,28))
                output_box1_detect, _ = model_cnn(box1_detect_tensor)
                pred_box1_detect = torch.max(output_box1_detect, 1)[1].data.numpy().squeeze()
                print('mnist model: {}'.format(pred_box1_detect))

                cv2.imshow("detect",frame_detect)

                cv2.drawContours(frame,[box_array[pos_box_min]],0,(0,255,0),10)

                
        
        cv2.imshow("gray", frame_bin)

        frame_box1_resized = cv2.resize(gray_frame_box1, (28,28), interpolation = cv2.INTER_AREA)
        frame_box2_resized = cv2.resize(gray_frame_box2, (28,28), interpolation = cv2.INTER_AREA)
        
        cv2.rectangle(frame,(box1_x-int(box1_w/2),box1_y+int(box1_h/2)),(box1_x+int(box1_w/2),box1_y-int(box1_h/2)), (0,255,0),2)
        cv2.rectangle(frame,(box2_x-int(box2_w/2),box2_y+int(box2_h/2)),(box2_x+int(box2_w/2),box2_y-int(box2_h/2)), (0,255,0),2)


        thresh = 200
        frame_box1_resized = cv2.threshold(frame_box1_resized, thresh, 255, cv2.THRESH_BINARY)[1]
        frame_box1_resized = cv2.bitwise_not(frame_box1_resized)  
        frame_box2_resized = cv2.threshold(frame_box2_resized, thresh, 255, cv2.THRESH_BINARY)[1]
        frame_box2_resized = cv2.bitwise_not(frame_box2_resized)

        M = cv2.getRotationMatrix2D((frame_box1.shape[1]//2,frame_box1.shape[0]//2),5,1) #3 grados
        frame_box1 = cv2.warpAffine(frame_box1,M,(frame_box1.shape[1],frame_box1.shape[0]))
       
        
        #print(pytesseract.image_to_string(frame_box1,config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'))

        #box1_tensor = transform(frame_box1_resized) #cnv img a tensor
        #box2_tensor = transform(frame_box2_resized) #cnv img a tensor
       

        #box1_tensor = torch.reshape(box1_tensor, (1,1,28,28))
        #box2_tensor = torch.reshape(box2_tensor, (1,1,28,28))


        #output_box1, _ = model_cnn(box1_tensor)
        #output_box2, _ = model_cnn(box2_tensor)
        #pred_box1 = torch.max(output_box1, 1)[1].data.numpy().squeeze()
        #pred_box2 = torch.max(output_box2, 1)[1].data.numpy().squeeze()

        #print('BOX 1: {}, BOX 2: {}'.format(pred_box1,pred_box2))
       
        #print(output_box1)

        #cv2.imwrite("frame.png",frame_copy)

        #frame_box1_resized = cv2.resize(frame_box1_resized, (280,280), interpolation = cv2.INTER_AREA)
        #frame_box2_resized = cv2.resize(frame_box2_resized, (280,280), interpolation = cv2.INTER_AREA)



        cv2.imshow("Video", frame)
        cv2.imshow("box1",frame_box1_resized)
        cv2.imshow("box2",frame_box2_resized)

        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break