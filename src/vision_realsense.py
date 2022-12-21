
import pyrealsense2 as rs
import numpy as np
import cv2

#PARAMETER
LIDAR_DISTANCE = 5
MERGE_CONSTANTA = 10
MIN_OBS_AREA = 1000

#THRESSHOLD
obs_thress = [0,0,0,180,255,50]
ball_thress = [0,150,60,7,255,255]

#obstacle buuffer and ball buffer
all_obs = [] 
ball_xyz = []




# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    # print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

JARAK_TOLERAN = LIDAR_DISTANCE / depth_scale

#Create align for syncronized wide between streo and rgb 
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Build a pipeline 
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)


        #GET EGB and Streo stream from pipeline
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        print(aligned_depth_frame)
        # print(color_frame)

        #INTRINSIK ini digunakan untuk parameter nanti
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        #safety apabila frame kosong
        if not aligned_depth_frame or not color_frame:
            continue

        #Konversi dari data kamera menjadi sebuah matrik gambar
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # print(aligned_depth_frame.get_data())
        print(depth_image)
        color_image = np.asanyarray(color_frame.get_data())

        #WHite buffer untuk buffer hapus bg
        white_buffer = np.ones((depth_image.shape[0],depth_image.shape[1],3)) * 255

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > JARAK_TOLERAN) | (depth_image_3d <= 0), 0, white_buffer)
        
        


        #OBSTACLE DETECTION 3D
        color_image_hsv = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
        buffer_obs = cv2.inRange(color_image_hsv,(obs_thress[0],obs_thress[1],obs_thress[2]),(obs_thress[3],obs_thress[4],obs_thress[5]))
        obstacle_final = cv2.bitwise_and(bg_removed,bg_removed,mask=buffer_obs)
        obstacle_final =obstacle_final.astype(np.uint8)
        obstacle_final = cv2.cvtColor(obstacle_final,cv2.COLOR_BGR2GRAY)
        obstacle_final = cv2.morphologyEx(obstacle_final, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_CONSTANTA,MERGE_CONSTANTA)))
        obstacle_final =cv2.erode(obstacle_final,(15,15))
        contours, hierarchy = cv2.findContours(image=obstacle_final, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if contours!=0:
            for con in contours:
                area = cv2.contourArea(con)
                if area > MIN_OBS_AREA:
                    x,y,w,h = cv2.boundingRect(con)
                    cv2.rectangle(color_image,(x,y),(x+w,y+h),(255,0,255),1)
                    titik_tengah = [int(x+(w/2)),int(y+(h/2))]
                    cv2.circle(color_image,(titik_tengah[0],titik_tengah[1]),2,(255,255,255),2)

                    #FIND DISTANCE
                    jarak_obs = aligned_depth_frame.get_distance(titik_tengah[0],titik_tengah[1])
                    cv2.putText(color_image,f"OBS : {round(jarak_obs,2)} ",(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)
                    real_obs_coordinates = rs.rs2_deproject_pixel_to_point(depth_intrin, [titik_tengah[0], titik_tengah[1]], jarak_obs)

                    #add obs to all obs
                    all_obs.append(real_obs_coordinates)

        

        #BALL DETECTION
        ball_thresshold = cv2.inRange(color_image_hsv,(ball_thress[0],ball_thress[1],ball_thress[2]),(ball_thress[3],ball_thress[4],ball_thress[5]))
        ball_thresshold = cv2.bitwise_and(bg_removed,bg_removed,mask=ball_thresshold)
        ball_thresshold =ball_thresshold.astype(np.uint8)
        ball_thresshold = cv2.cvtColor(ball_thresshold,cv2.COLOR_BGR2GRAY)
        ball_thresshold = cv2.morphologyEx(ball_thresshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))


        contours, hierarchy = cv2.findContours(image=ball_thresshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        # print(contours)
        if contours:
            ball_contour = max(contours, key = cv2.contourArea)
            ball_center,r_ball = cv2.minEnclosingCircle(ball_contour)
            # print(int(ball_center))
            cv2.circle(color_image,(int(ball_center[0]),int(ball_center[1])),int(r_ball),(0,255,255),2)
            jarak_bola = aligned_depth_frame.get_distance(int(ball_center[0]),int(ball_center[1]))
            cv2.putText(color_image,f"BALL : {round(jarak_bola,2)} ",(int(ball_center[0] - r_ball),int(ball_center[1] - int(r_ball+20))),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)

            #REAL WORD XYZ
            ball_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(ball_center[0]), int(ball_center[1])], jarak_bola)

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow('Depth', obstacle_final)
        cv2.imshow('Color', color_image)


        # print(ball_xyz)
        all_obs.clear()
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
    
finally:
    pipeline.stop()