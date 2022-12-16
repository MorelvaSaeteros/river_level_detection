from __future__ import division
from cProfile import label
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable

import pandas as pd
from joblib import load

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="object confidence threshold")
    parser.add_argument("--webcam", type=int, default=0,  help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--directorio_video", type=str, default='database/video_out/araxes_06-04-2022.mp4', help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, default='checkpoints/yolov3_ckpt_99.pth', help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)


    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam==1:
        cap = cv2.VideoCapture(0)
        out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,960))
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
        # cap.set(1,1350)
        print(opt.directorio_video)

        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        # out = cv2.VideoWriter('outp.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,760))
        out = cv2.VideoWriter('outp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1470,760)) 

    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]

   
    df = pd.read_csv('database/csv/araxes_06-04-2022.csv')
    cnt = 0

    df_db = pd.read_csv('database/csv_db/araxes_db_06-04-2022.csv')

    model_load = load('model_araxes_0.58685.pkl')

    level_ref = 0.456
    level_px_ref = 399
    # level_px_ref = round(500/760,5)

    res = pd.DataFrame(columns=['level_px','level_model','level'])
    
    while cap:
        ret, frame = cap.read()
        if ret is True:

            frame = cv2.resize(frame, (1280, 760), interpolation=cv2.INTER_CUBIC)

            original = frame.copy()

            RGBimg=Convertir_RGB(frame)
            imgTensor = transforms.ToTensor()(RGBimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(Tensor))


            with torch.no_grad():
                detections = model(imgTensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            points = []
            x = []
            y = []

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        color = [int(c) for c in colors[int(cls_pred)]]
                        #print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                        #frame = cv2.rectangle(frame, (int(x1), int(y1 + box_h)), (int(x2), int(y1)), color, 5)
                        frame = cv2.circle(frame, (int(x1 + box_w/2), int(y1 + box_h/2)),2, (0,255,0), 5)
                        #cv2.putText(frame, classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)# Nombre de la clase detectada
                        #cv2.putText(frame, str("%.2f" % float(conf)), (int(x2), int(y2 - box_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 5) # Certeza de prediccion de la clase

                        points.append([int(x1 + box_w/2), int(y1 + box_h/2)])
                        x.append(int(x1 + box_w/2))
                        y.append(int(y1 + box_h/2))

            points = sorted(points)

            # fps = cap.get(cv2.CAP_PROP_FPS)
            # print("Frames per second using video.get(cv2.CAP_PROP_FPS): {0}".format(fps))

            if len(x)>1:
                linear_model=np.polyfit(x,y,1)
                linear_model_fn=np.poly1d(linear_model)

                y_start = linear_model_fn(points[0][0])
                y_end = linear_model_fn(points[len(points)-1][0])
                level_px_0 = linear_model_fn(frame.shape[1]/2-200)
                level_px_1 = linear_model_fn(frame.shape[1]/2-50)
                level_px_2 = linear_model_fn(frame.shape[1]/2+50)
                level_px_3 = linear_model_fn(frame.shape[1]/2+200)

                df_hour = df.iloc[cnt]['date']
                df_hour = df_hour.split(' ')[1]
                nivel = 0

                for i,hour in enumerate(df_db['Hora']):
                    if hour[:4] == df_hour[:4]:
                        nivel = df_db.iloc[i]['nivel']


                df.iloc[cnt]= (df.iloc[cnt]['date'], int(level_px_0), int(level_px_1), int(level_px_2), int(level_px_3), nivel)
                cnt += 1

                hora = df_hour.split(':')
                hora = int(hora[0])*60 + int(hora[1])
                pd_level_px = pd.DataFrame([{hora,level_px_0,level_px_1,level_px_2,level_px_3}], columns=['date','level_px_0','level_px_1','level_px_2','level_px_3'])
                level_river_model = round(model_load.predict(pd_level_px)[0]/1000,3)

                level_river_0 = round((level_px_0*level_ref)/level_px_ref,3)
                level_river_1 = round((level_px_1*level_ref)/level_px_ref,3)
                level_river_2 = round((level_px_2*level_ref)/level_px_ref,3)
                level_river_3 = round((level_px_3*level_ref)/level_px_ref,3)

                mean_level = round((level_river_0+level_river_1+level_river_2+level_river_3)/4,3)

                res.loc[len(res)] = [mean_level, level_river_model, nivel]


                dif_level_0 = abs(round(level_river_0 - nivel,3))
                dif_level_1 = abs(round(level_river_1 - nivel,3))
                dif_level_2 = abs(round(level_river_2 - nivel,3))
                dif_level_3 = abs(round(level_river_3 - nivel,3))

                err_rel_0 = round((dif_level_0/nivel)*100,2)
                err_rel_1 = round((dif_level_1/nivel)*100,2)
                err_rel_2 = round((dif_level_2/nivel)*100,2)
                err_rel_3 = round((dif_level_3/nivel)*100,2)


                zoom_level_px_0 = original[int(level_px_0)-20:int(level_px_0)+20,int(frame.shape[1]/2)-220:int(frame.shape[1]/2)-180]
                zoom_level_px_1 = original[int(level_px_1)-20:int(level_px_1)+20,int(frame.shape[1]/2)-70:int(frame.shape[1]/2)-30]
                zoom_level_px_2 = original[int(level_px_2)-20:int(level_px_2)+20,int(frame.shape[1]/2)+30:int(frame.shape[1]/2)+70]
                zoom_level_px_3 = original[int(level_px_3)-20:int(level_px_3)+20,int(frame.shape[1]/2)+180:int(frame.shape[1]/2)+220]
                zoom_level_px_0 = cv2.resize(zoom_level_px_0, (190, 190), interpolation=cv2.INTER_CUBIC)
                zoom_level_px_1 = cv2.resize(zoom_level_px_1, (190, 190), interpolation=cv2.INTER_CUBIC)
                zoom_level_px_2 = cv2.resize(zoom_level_px_2, (190, 190), interpolation=cv2.INTER_CUBIC)
                zoom_level_px_3 = cv2.resize(zoom_level_px_3, (190, 190), interpolation=cv2.INTER_CUBIC)
                cv2.circle(zoom_level_px_0, (int(zoom_level_px_0.shape[1]/2), int(zoom_level_px_0.shape[0]/2)), 2, (0,0,255), 5)
                cv2.circle(zoom_level_px_1, (int(zoom_level_px_1.shape[1]/2), int(zoom_level_px_1.shape[0]/2)), 2, (0,0,255), 5)
                cv2.circle(zoom_level_px_2, (int(zoom_level_px_2.shape[1]/2), int(zoom_level_px_2.shape[0]/2)), 2, (0,0,255), 5)
                cv2.circle(zoom_level_px_3, (int(zoom_level_px_3.shape[1]/2), int(zoom_level_px_3.shape[0]/2)), 2, (0,0,255), 5)
                cv2.putText(zoom_level_px_0, 'level_px_0', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(zoom_level_px_1, 'level_px_1', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(zoom_level_px_2, 'level_px_2', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(zoom_level_px_3, 'level_px_3', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                v_zomm = cv2.vconcat([zoom_level_px_0, zoom_level_px_1, zoom_level_px_2, zoom_level_px_3])

                cv2.circle(frame, (points[0][0], int(y_start)), 2, (0,0,0), 5)
                cv2.circle(frame, (points[len(points)-1][0], int(y_end)), 2, (0,0,0), 5)
                cv2.line(frame, (points[0][0], int(y_start)), (points[len(points)-1][0], int(y_end)), (0,0,255), 3)

                cv2.polylines(frame, np.int32([points]), False, (255,255,255), 2)

                cv2.circle(frame, (int(frame.shape[1]/2)-200, int(level_px_0)), 2, (255,0,0), 8)
                cv2.circle(frame, (int(frame.shape[1]/2)-50, int(level_px_1)), 2, (255,0,0), 8)
                cv2.circle(frame, (int(frame.shape[1]/2)+50, int(level_px_2)), 2, (255,0,0), 8)
                cv2.circle(frame, (int(frame.shape[1]/2)+200, int(level_px_3)), 2, (255,0,0), 8)
                
                text_y = 480
                cv2.putText(frame, "Puntos: {}".format(len(points)), (10, 30 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "Tendencia: {}".format(linear_model_fn), (10, 60 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "PX       Predict    Real    Diff   error[%]   Mean Predict", (180, 90 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "Level 0: {}".format(int(level_px_0)), (10, 120 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "Level 1: {}".format(int(level_px_1)), (10, 150 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "Level 2: {}".format(int(level_px_2)), (10, 180 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "Level 3: {}".format(int(level_px_3)), (10, 210 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(level_river_0), (350, 120 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(level_river_1), (350, 150 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(level_river_2), (350, 180 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(level_river_3), (350, 210 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(nivel), (500, 170 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, "{}".format(dif_level_0), (620, 120 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(dif_level_1), (620, 150 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(dif_level_2), (620, 180 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(dif_level_3), (620, 210 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(err_rel_0), (770, 120 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(err_rel_1), (770, 150 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(err_rel_2), (770, 180 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(err_rel_3), (770, 210 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(mean_level), (930, 120 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, "Model Predict", (930, 180 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, "{}".format(round(level_river_model,3)), (930, 210 + text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
                cv2.putText(frame, "Hora DB: {}0".format(df_hour[:4]), (960, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            
                Convertir_BGR(RGBimg)
                
                h_out = cv2.hconcat([RGBimg, v_zomm])

                out.write(h_out)
                cv2.imshow('frame', h_out)
            else:
                cv2.imshow('frame', original)


            if cv2.waitKey(30) & 0xFF == ord('q'):
                res.to_csv('database/csv/result_araxes_06-04-2022.csv', index=False)
                pd.plotting.scatter_matrix(res, figsize = (10, 10))
                plt.show()

                # fig, axs = plt.subplots(figsize=(12, 4))
                # res.plot.area(ax=axs)
                # plt.show()

                res.plot.area(figsize=(12, 4), subplots=True)
                plt.show()

                res.plot()
                plt.show()
                break
        else:
            df.to_csv('database/csv/araxes_06-04-2022.csv', index=False)
            res.to_csv('database/csv/result_araxes_06-04-2022.csv', index=False)
            pd.plotting.scatter_matrix(res, figsize = (10, 10))
            plt.show()

            # fig, axs = plt.subplots(figsize=(12, 4))
            # res.plot.area(ax=axs)
            # plt.show()

            res.plot.area(figsize=(12, 4), subplots=True)
            plt.show()

            res.plot()
            plt.show()
            break
        


    out.release()
    cap.release()
    cv2.destroyAllWindows()
