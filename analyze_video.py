import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# path_db_video = './database/video'
# list_videos = []

# files = os.listdir(path_db_video)
# print("Hay {} fechas disponibles".format(len(files)))
# print("Videos por fecha:")
# for video_file in files:
#     video = os.listdir(path_db_video + '/' + video_file)
#     print("{} ---> {}".format(len(video), video_file))
#     for video_name in video:
#         list_videos.append(VideoFileClip(path_db_video + '/' + video_file + '/' + video_name))

    # final_clip = concatenate_videoclips(list_videos)
    # final_clip.write_videofile('database/video_out/{}.mp4'.format(video_file))

cap = cv2.VideoCapture('./database/video/araxes_06-04-2022/out_06-04-2022_20h_01m_36s.mp4')
cap.set(1,5670) #1340 5860 3640

while True:
    scc, frame = cap.read()
    if scc:
        frame = cv2.resize(frame, (1280, 760), interpolation=cv2.INTER_CUBIC)
        cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break