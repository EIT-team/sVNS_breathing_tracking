import cv2
import numpy as np
# import easyocr
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=0.5, fs=30, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b,a = butter(order,normal_cutoff,btype='low',analog=False)
    return filtfilt(b,a,data)

def is_filter_stable(b,a):
    poles = np.roots(a) # compute poles as roots of the denomintor a
    return np.all(np.abs(poles) < 1)

def select_roi(frame):
    """
    Pauses the video to let the user select a region of interest (ROI).
    """
    cv2.namedWindow("Select ROI", cv2.WINDOW_KEEPRATIO)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return roi

# def write_to_csv(path_to_csv,detected_data):
#     with open(path_to_csv, 'w', newline='') as csvfile:
#         fieldnames = ['Time (s)'] + ['Detected']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         # shape rows correctly
#         row = {'Time (s)': None, 'Detected': None}
#         for time in detected_data.keys():
#             row['Time (s)'] = time
#             row['Area'] = detected_data[time]
#             writer.writerow(row)

def extract_RR(video_path):
    global fps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = cv2.TrackerCSRT_create()    
    frame_count = 0
    frame_skip = 1
    roi = None
    cv2.namedWindow("Video Processing", cv2.WINDOW_KEEPRATIO)
    target_width = 1280
    areas = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read video.")
            cap.release()
            break
        # Rotate if there's an issue with the smartphone metadata interpretation
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        H, W, _ = frame.shape # get the frame dimensions
        
        if frame_count == 0:
            # Show the first frame and let the user select the region of interest
            print("Displaying first frame for ROI selection...")
            roi = select_roi(frame)
            tracker.init(frame, roi)  # Initialize tracker with ROI
            x, y, w, h = roi
            print("ROI selected. Starting video processing...")
        success, roi = tracker.update(frame)
        if success:
            x,y,w,h = map(int,roi)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            frame_roi = frame[y:y+h, x:x+w]
            #hsv = cv2.cvtColor(frame[y:y+h, x:x+w],cv2.COLOR_BGR2HSV)
            hsv = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2HSV)
            # Original values:
            # lower_green = np.array([35,50,50])
            # upper_green = np.array([85,255,255])

            #Experimental
            lower_green = np.array([35,50,50])
            upper_green = np.array([90,255,255])

            mask = cv2.inRange(hsv,lower_green,upper_green)
            green_region = cv2.bitwise_and(frame_roi,frame_roi,mask=mask)
            gray = cv2.cvtColor(green_region, cv2.COLOR_BGR2GRAY)
            
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
            # Normalize and threshold to get binary edge image
            edges = np.uint8(255 * edges / np.max(edges))
            
            _, edge_binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
            sobel_edges_colored = cv2.merge([edge_binary, edge_binary, edge_binary])
            edges_black = cv2.bitwise_not(sobel_edges_colored)
            frame[y:y+h, x:x+w] = cv2.bitwise_and(frame[y:y+h, x:x+w], edges_black)
            # Calculate edge metric (e.g., total edge pixel count)
            edge_metric = np.sum(edge_binary > 0)
            areas.append(edge_metric)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                #timestamps.append(time.time())
            timestamps.append(timestamp) # ms / 1000 => seconds
           
                # Adjust contour coordinates to match full frame
               # max_contour_shifted = max_contour + np.array([x, y])  # Shift contour back to full-frame coordinates

                # cv2.drawContours(frame[y:y+h, x:x+w],[max_contour],-1,(0,255,0),2)
                # cv2.putText(frame[y:y+h, x:x+w],f"Area = {area}",(x,y+h+10),cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5,(0,255,0),2)

                # cv2.drawContours(frame, [max_contour_shifted], -1, (0, 255, 0), 2)
            
            text_position = (x, y - 10 if y > 20 else y + h + 20)  # Adjust if near top edge
            cv2.putText(frame, f"Area = {edge_metric:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Timestamp: {timestamp: .2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # expandVal = 40
            # x1 = max(0, x-expandVal)
            # y1 = max(0, y-expandVal)
            # w1 = min(W, w+expandVal)
            # h1 = min(H, h+expandVal)

            # Zoom in on the region of interest
            #zoomed_frame = frame[y1:y+h1, x1:x+w1]
            #zoomed_frame = frame[y-20:y+h+20, x-20:x+w+20]
            # Perform OCR on the zoomed frame
           
                # Draw results on the frame
            # cv2.putText(frame, f"{text} ({confidence:.2f})", (x, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # old_gray = frame_gray.copy()
            # roi_points = new_points
        # if tracker update unsuccessful
        else:
            print("Tracking lost. Re-select ROI...")
            roi = select_roi(frame)
            #tracker.init(frame, roi)  # Initialize tracker with ROI
            #x, y, w, h = roi
            if roi and all(v > 0 for v in roi):
                tracker = cv2.TrackerCSRT_create()  # Reinitialize tracker
                tracker.init(frame, roi)
            else:
                print("Invalid ROI selected, skipping this frame...")
                continue  # Skip processing this frame
            
            print("ROI selected. Starting video processing...")
        
        # Display the current frame
        aspRatio = W / H
        new_height = int(target_width / aspRatio)
        frame = cv2.resize(frame, (target_width, new_height))
        cv2.imshow("Video Processing", frame)
        frame_count += frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            print("Manual adjustment of ROI...")
            roi = select_roi(frame)
            tracker = cv2.TrackerCSRT_create()  # Reinitialize tracker with CSRT
            tracker.init(frame, roi)  # Initialize tracker with ROI
            x, y, w, h = roi
            print("ROI selected. Continuing video processing...")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    timestamps = np.array(timestamps) # convert for the normalistaion
    #timestamps -= timestamps[0]
    return areas, timestamps


if __name__ == "__main__":
    #video_path = r"C:\Users\erutkovs\OneDrive - University College London\MRes sVNS project\Human trial\human_trial_recordings\data_06012025_pat_14\video\Human 014 060125\014_sVNS_C_1.6mA 1ms 20Hz 30s~3.mp4"  # Replace with the path to your video file
    #video_path = r"../../data_06012025_pat_14\video\Human 014 060125\014_sVNS_C_1.6mA 1ms 20Hz 30s~3.mp4"
    video_path = "../data_06012025_pat_14/video/Human 014 060125/014_sVNS_P_900uA 1ms 20Hz 30s.mp4"
    csv_path = "../data_06012025_pat_14/video/processed/014_area_detection_test_CSRT.csv"
    areas, timestamps = extract_RR(video_path)
    #print("Final Detected Data:", detected_data)
    #write_to_csv("../../data_06012025_pat_14/video/processed/014_area_detection_test.csv", areas)
    df = pd.DataFrame({'Time (s)': timestamps, 'Area': areas})
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    print(f"Video FPS: {fps:.2f}")
    # Visualise
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(timestamps,areas,marker='o',linestyle='-',color='b',label='Area change')
    plt.xlabel("Time (s)")
    plt.ylabel("Detected Area")
    plt.title("Area Change Over Time")
    plt.legend()
    plt.grid(True)

    # smoothed areas
    
    plt.subplot(2,1,2)
    
    order = 4
    cutoff = 2
    fs = 30
    normal_cutoff = cutoff / (fs/2)
    b,a = butter(order,normal_cutoff,btype='low',analog=False)
    print("Filter is stable? ", is_filter_stable(b,a))

    smoothed_areas = butter_lowpass_filter(areas, cutoff, fs, order)
    plt.plot(timestamps,smoothed_areas,linestyle='-', linewidth=2, color='b', label='Smoothed Area')
    plt.xlabel("Time (s)")
    plt.ylabel("Detected Area")
    plt.title("Smoothed Area Change Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()