import cv2
import numpy as np
import easyocr
import csv

def select_roi(frame):
    """
    Pauses the video to let the user select a region of interest (ROI).
    """
    cv2.namedWindow("Select ROI", cv2.WINDOW_KEEPRATIO)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return roi

def write_to_csv(path_to_csv,detected_data):
    with open(path_to_csv, 'w', newline='') as csvfile:
        fieldnames = ['Time (s)'] + ['Detected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # shape rows correctly
        row = {'Time (s)': None, 'Detected': None}
        for time in detected_data.keys():
            row['Time (s)'] = time
            row['Detected'] = detected_data[time]
            writer.writerow(row)

def extract_RR(video_path):
    cap = cv2.VideoCapture(video_path)
    tracker = cv2.TrackerKCF_create()    
    frame_count = 0
    frame_skip = 1
    roi = None
    detected_data = {}
    cv2.namedWindow("Video Processing", cv2.WINDOW_KEEPRATIO)
    target_width = 1280

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read video.")
            cap.release()
            break
        # Rotate if there's an issue with the smartphone metadata interpretation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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
            tracker.init(frame, roi)  # Initialize tracker with ROI
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
            tracker.init(frame, roi)  # Initialize tracker with ROI
            print("ROI selected. Continuing video processing...")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_data


if __name__ == "__main__":
    #video_path = r"C:\Users\erutkovs\OneDrive - University College London\MRes sVNS project\Human trial\human_trial_recordings\data_06012025_pat_14\video\Human 014 060125\014_sVNS_C_1.6mA 1ms 20Hz 30s~3.mp4"  # Replace with the path to your video file
    #video_path = r"../../data_06012025_pat_14\video\Human 014 060125\014_sVNS_C_1.6mA 1ms 20Hz 30s~3.mp4"
    video_path = "../data_06012025_pat_14/video/Human 014 060125/014_sVNS_P_900uA 1ms 20Hz 30s.mp4"
    detected_data = extract_RR(video_path)
    #print("Final Detected Data:", detected_data)
    #write_to_csv("../../data_06012025_pat_14/video/processed/014_sVNS_P_900uA 1ms 20Hz 30s~2.csv", detected_data)