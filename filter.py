from turtle import color
import cv2
import numpy as np
import pytesseract
import torchvision.transforms as transforms
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from collections import Counter
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scikitplot as skplt


cap = cv2.VideoCapture("araxes1.mp4")
cap.set(1,2200); # Where frame_no is the frame you want

level_river_1 = 2530
level_river_2 = 2700
level_river_3 = 3100
level_river_4 = 3200

#parametros filtro
threshold_binary_min = 180 
threshold_binary_max = 190 
threshold_ruido = 1000 #cajas peq o grandes area en pixeles
box_detection = 5000

model_cnn = torch.load("model_cnn.pt")
model_cnn.eval()
transform = transforms.ToTensor()
transform_batch = transforms.Compose([
    ToTensor()
])

level_py = []
level_model = []

out_video_rgb = cv2.VideoWriter('out_rgb.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (2420,2160))
out_video_filter = cv2.VideoWriter('out_filter.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (3600,2400))

while True:

    

    scc, frame = cap.read()

    if scc:
        frame_copy = frame.copy()
        frame_filtro_1 = frame.copy()
        frame_filtro_2 = frame.copy()

        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)         
        binary = cv2.threshold(gray, threshold_binary_min, threshold_binary_max, cv2.THRESH_BINARY)[1]
        gauss = cv2.GaussianBlur(binary, (3,3), 0)

        median = cv2.medianBlur(gauss,5)
        median_original = median.copy()

        canny = cv2.Canny(median,50,60)
        
        (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            if area < threshold_ruido:             
                cv2.drawContours(median,[box],-1,(0,0,0),-1)
            else:
                cv2.drawContours(median,[box],0,(0,0,255),10)


        canny3 = cv2.Canny(median,50,60)
        
        (contornos3,_) = cv2.findContours(canny3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_cont = 0
        box_array = []
        for c3 in contornos3:
            area3 = cv2.contourArea(c3)
            rect3 = cv2.minAreaRect(c3)
            box3 = cv2.boxPoints(rect3)
            box3 = np.int0(box3)
            if area3 > box_detection:    
                box_cont += 1
                box_array.append(box3)               
                cv2.drawContours(frame_filtro_2,[box3],0,(255,0,255),10)

        


        if box_cont >= 1:
            box_arr_tmp = []
                    
            for coo in box_array:
                box_arr_tmp.append(coo[1][1])
            
            pos_box_min = box_arr_tmp.index(max(box_arr_tmp))

            cv2.rectangle(frame_filtro_2,(box_array[pos_box_min][1][0],box_array[pos_box_min][1][1]),(box_array[pos_box_min][3][0],box_array[pos_box_min][3][1]), (255,255,40),10)
            
            
            frame_detect = frame_filtro_2[box_array[pos_box_min][1][1]+30:box_array[pos_box_min][3][1]-5,box_array[pos_box_min][1][0]+90:box_array[pos_box_min][3][0]-9]
            frame_detect_model = frame_filtro_2[box_array[pos_box_min][1][1]+30:box_array[pos_box_min][3][1]-10,box_array[pos_box_min][1][0]+100:box_array[pos_box_min][3][0]-9]
            
            cv2.imwrite("roi.jpg",frame_detect)

            detect_pytess = pytesseract.image_to_string(frame_detect,config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
            cv2.putText(frame_filtro_2, "{}".format(detect_pytess.strip()), (500,500),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
            detect_pytess = detect_pytess.strip()
            if detect_pytess != '':
                if int(detect_pytess) <= 9:
                    level_py.append(int(detect_pytess))

            gray_frame_tensor = cv2.cvtColor(frame_detect_model, cv2.COLOR_BGR2GRAY)
            gray_frame_tensor = cv2.threshold(gray_frame_tensor, 200, 255, cv2.THRESH_BINARY)[1]
            #gray_frame_tensor = cv2.bitwise_not(gray_frame_tensor)  
            
            cv2.imwrite("roi_model.jpg",gray_frame_tensor)
            
            frame_box1_resized_detect = cv2.resize(gray_frame_tensor, (28,28), interpolation = cv2.INTER_AREA)
            box1_detect_tensor = transform(frame_box1_resized_detect)
            #print(box1_detect_tensor.shape)
            box1_detect_tensor = torch.reshape(box1_detect_tensor, (1,1,28,28))
            output_box1_detect, _ = model_cnn(box1_detect_tensor)
            pred_box1_detect = torch.max(output_box1_detect, 1)[1].data.numpy().squeeze()

            cv2.putText(frame_filtro_2, '{}'.format(pred_box1_detect), (500,400),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
            level_model.append(int(pred_box1_detect))

        x_model = []
        y_model = []
        max_model = 0
        x_py = []
        y_py = []
        max_py = 0

        counter_model = Counter(level_model)
        counter_py = Counter(level_py)

        for val_model in counter_model:
            x_model.append(val_model)
            y_model.append(counter_model[val_model])
        for val_py in counter_py:
            x_py.append(val_py)
            y_py.append(counter_py[val_py])


        plt.ion()
        plt.bar(x_model,y_model,color="green",label='Model')
        plt.bar(x_py,y_py,color="blue",label='Pytesseract')
        plt.ylabel('Repeticiones')
        plt.xlabel('Números identificados')
        plt.title('Repeticiones identificadas')
        plt.show()
        plt.savefig('Number_identifier.jpg')

     


        if len(y_model) > 0:
            max_model = x_model[y_model.index(max(y_model))]
        if len(y_py) > 0:
            max_py = x_py[y_py.index(max(y_py))]

        fps = cap.get(cv2.CAP_PROP_POS_FRAMES)

        cv2.line(frame_filtro_2,(0,int(frame_filtro_2.shape[0]/2)),(frame_filtro_2.shape[1],int(frame_filtro_2.shape[0]/2)),(255,255,255),3)

        cv2.putText(frame_filtro_2, str(fps), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        
        if fps < level_river_1:
            max_model = max_model/10
            max_py = max_py/10
        elif fps < level_river_2:
            max_model = (max_model/10)+1 
            max_py = (max_py/10)+1 
        elif fps < level_river_3:
            max_model = (max_model/10)+2
            max_py = (max_py/10)+2
        elif fps < level_river_4:
            max_model = (max_model/10)+3 
            max_py = (max_py/10)+4

        cv2.putText(frame_filtro_2, 'Model: ', (50,400),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
        cv2.putText(frame_filtro_2, 'Pytesseract', (50,500),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
        cv2.putText(frame_filtro_2, 'Number', (500,300),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
        cv2.putText(frame_filtro_2, 'Level', (800,300),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
        cv2.putText(frame_filtro_2, '{}'.format(max_model), (800,400),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
        cv2.putText(frame_filtro_2, '{}'.format(max_py), (800,500),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)


        median_original = cv2.resize(median_original, (1200, 1200), interpolation= cv2.INTER_LINEAR)
        median = cv2.resize(median, (1200, 1200), interpolation= cv2.INTER_LINEAR)
        canny = cv2.resize(canny, (1200, 1200), interpolation= cv2.INTER_LINEAR)
        canny3 = cv2.resize(canny3, (1200, 1200), interpolation= cv2.INTER_LINEAR)
        binary = cv2.resize(binary, (1200, 1200), interpolation= cv2.INTER_LINEAR)
        gray = cv2.resize(gray, (1200, 1200), interpolation= cv2.INTER_LINEAR)

        cv2.putText(gray, "Gray", (850,1100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(binary, "Binary", (850,1100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(median_original, "Median filter 1", (450,1100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(median, "Median filter 2", (450,1100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(canny, "Canny filter 1", (450,1100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        cv2.putText(canny3, "Canny filter 2", (450,1100),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)


        out_v = cv2.hconcat([gray, median_original, canny])
        out_v2 = cv2.hconcat([binary,median, canny3])
        out = cv2.vconcat([out_v,out_v2])

        roi_filter_read = cv2.imread('roi.jpg')
        roi_model_read = cv2.imread('roi_model.jpg')
        roi_filter_read = cv2.resize(roi_filter_read, (500, 1080), interpolation= cv2.INTER_LINEAR)
        roi_model_read = cv2.resize(roi_model_read, (500, 1080), interpolation= cv2.INTER_LINEAR)

        roi_concat = cv2.vconcat([roi_model_read,roi_filter_read])
        out_rgb = cv2.vconcat([frame_copy,frame_filtro_2])
        out_rgb_roi = cv2.hconcat([out_rgb,roi_concat])

        cv2.imwrite("filter.jpg", out)
        cv2.imwrite("out_rgb.jpg", out_rgb)

        cv2.imshow('out rgb',out_rgb_roi)
        cv2.imshow('filter',out)

        filter_rgb = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)

        out_video_rgb.write(out_rgb_roi)
        out_video_filter.write(filter_rgb)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


out_video_rgb.release()
out_video_filter.release()
cap.release()
cv2.destroyAllWindows()

    #sobel varianza para conv img energia --- luego gradiente --- desplazamiento temporal (digital imagen correlation) --- dinamyc programmig(optimizacion global)--- 