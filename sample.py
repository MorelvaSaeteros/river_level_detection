import cv2
import pandas as pd
import os

path_db_video = './database/video'

set_frame = 0
set_frame_0 = 1340#3350  #1340
set_frame_1 = 5860#1100  #5860
set_frame_2 = 3640#5670  #3640
fact_set_frame_0 = 130
fact_set_frame_1 = 150
fact_set_frame_2 = 160

time_frame = 60
dist_frame = 6770
cnt_video = 0

cnt_labels = 0
df_db = pd.read_csv('database/csv_db/araxes_db_06-04-2022.csv')
level_river = []

data = pd.DataFrame(columns=['date','level_px_0','level_px_1','level_px_2','level_px_3','level'])

files = os.listdir(path_db_video)
print("Hay {} fechas disponibles".format(len(files)))
print("Videos por fecha:")
for video_file in files:
    video = os.listdir(path_db_video + '/' + video_file)
    print("{} ---> {}".format(len(video), video_file))
    video.sort()
    print(video)
    out = cv2.VideoWriter('database/video_out/{}.mp4'.format(video_file), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280,760))

    for video_name in video:
        cap = cv2.VideoCapture(path_db_video + '/' + video_file + '/' + video_name)
        if cap.isOpened():
            print("Video: {}".format(video_name))
            print("Frame: {}".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            date = video_name.split('_')[1]
            hour = int(video_name.split('h_')[0][-2:])
            min_ = int(video_name.split('h_')[1][:2])

            cnt_frame = 0

            if cnt_video == 0:
                set_frame = set_frame_0
            elif cnt_video == 1:
                set_frame = set_frame_1
            elif cnt_video == 2:
                set_frame = set_frame_2  
            elif cnt_video == 3:
                set_frame_0 = set_frame_0 + fact_set_frame_0
                set_frame_1 = set_frame_1 + fact_set_frame_1
                set_frame_2 = set_frame_2 + fact_set_frame_2  
                set_frame = set_frame_0

                cnt_video = 0


            cap.set(1,set_frame)

            while True:
                scc, frame = cap.read()
                if scc:
                    frame = cv2.resize(frame, (1280, 760), interpolation=cv2.INTER_CUBIC)
                    original = frame.copy()
                    cv2.putText(original, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    cv2.putText(original, str(video_name), (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    cv2.putText(original, str(cnt_video), (20,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % time_frame == 0:
                        cnt_frame += 1
                        cap.set(1,set_frame+cnt_frame*dist_frame)
                        min_ = min_ + 4

                    if min_ > 60:
                        min_ = 0
                        hour += 1

                    if min_ < 10:
                        min_tmp = '0' + str(min_)
                    else:
                        min_tmp = min_

                    data.loc[len(data)] = ["{} {}:{}".format(date,hour,min_tmp),0,0,0,0,0]

                    cv2.rectangle(original, (150,350), (1200,750), (0,255,0), thickness=5)
                    cv2.circle(original, (675,550), 5, (255,0,0), thickness=5)
                    x_label = round(675/original.shape[1],6)
                    y_label = round(550/original.shape[0],6)
                    w_label = round(1050/original.shape[1],6)
                    h_label = round(400/original.shape[0],6)
                    
                    caudal = 0
                    hour_tmp = "{}:{}".format(hour,min_tmp)[:4]
                    for i,hour_ in enumerate(df_db['Hora']):
                        if hour_[:4] == hour_tmp:
                            caudal = df_db.iloc[i]['caudal']
                    
                    
                    level_river.append(caudal)
                    level_river_tmp = list(set(level_river)) # elimina duplicados
                    level_river_tmp.sort()

                    pos_caudal = level_river_tmp.index(caudal)

                    file = open("database/data/labels/{}.txt".format(cnt_labels), "w")
                    file.write("{} {} {} {} {} \n".format(pos_caudal,x_label,y_label,w_label,h_label))
                    file.close()
                    
                    cv2.putText(original, "Clases etiquetadas: {}".format(len(level_river_tmp)), (20,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                    cv2.putText(original, "{}".format(level_river_tmp), (20,230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                    file_classes = open("database/data/labels/classes.txt", "w")
                    for classes in level_river_tmp:
                        file_classes.write("{} \n".format(str(classes)))
                    file_classes.close()

                    cv2.imwrite("database/data/images/{}.jpg".format(cnt_labels), frame)

                    out.write(frame)
                    cv2.imshow('frame', original)

                    cnt_labels += 1
                    
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                else:
                    data.to_csv('database/csv/{}.csv'.format(video_file),index=False)
                    break
            

            cnt_video += 1
            


            print()

# cap = cv2.VideoCapture('./videos/araxes_04.mp4')

# date = '06-04-2022'
# hour = 19
# min_ = 32

# cap.set(1,set_frame)

# cnt = 0
# cnt_frame = 0



# while True:

#     scc, frame = cap.read()

#     if scc:
#         frame = cv2.resize(frame, (1280, 760), interpolation=cv2.INTER_CUBIC)

#         cnt += 1

#         if cnt % time_frame == 0:
#             cnt_frame += 1
#             cap.set(1,set_frame+cnt_frame*dist_frame)
#             cnt = 0
#             min_ = min_ + 4

#         data.loc[len(data)] = ["{} {}:{}".format(date,hour,min_),0,0,0,0,0]

#         out.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break
#     else:

#         data.to_csv('database/csv/araxes_{} {}-{}.csv'.format(date,hour,min_),index=False)

#         break