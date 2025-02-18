import cv2
import numpy as np
import easyocr
import csv

def select_roi(frame):
    """
    Pauses the video to let the user select a region of interest (ROI).
    """
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

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    reader = easyocr.Reader(['en'])
    
    frame_count = 0
    frame_skip = 1
    roi = None
    tracker = None
    detected_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read video.")
            cap.release()
            break
        
        H, W, _ = frame.shape # get the frame dimensions

        # Rotate if there's an issue with the smartphone metadata interpretation
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame_count == 0:
            # Show the first frame and let the user select the region of interest
            print("Displaying first frame for ROI selection...")
            roi = select_roi(frame)
            x, y, w, h = roi
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            roi_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32).reshape(-1, 1, 2)
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # Initialize the tracker
            #tracker = cv2.TrackerKCF_create()
            # tracker = cv2.TrackerKCF_create()
            # tracker.init(frame, roi)
            print("ROI selected. Starting video processing...")
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, roi_points, None, **lk_params)
        # Update the tracker for subsequent frames
        #success, roi = tracker.update(frame)
        if status:
            #x, y, w, h = map(int, roi)
            new_x, new_y = new_points[0][0]
            x, y = int(new_x - w // 2), int(new_y - h // 2)
            # expandVal = 40
            # x1 = max(0, x-expandVal)
            # y1 = max(0, y-expandVal)
            # w1 = min(W, w+expandVal)
            # h1 = min(H, h+expandVal)

            # Zoom in on the region of interest
            #zoomed_frame = frame[y1:y+h1, x1:x+w1]
            #zoomed_frame = frame[y-20:y+h+20, x-20:x+w+20]
            # Perform OCR on the zoomed frame
            ocr_results = reader.readtext(frame[y:y+h,x:x+w], batch_size=8, allowlist='0123456789')
            ocr_results = [result for result in ocr_results if result[2] >= 0.9]
            for result in ocr_results:
                text, confidence = result[1], result[2]
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                detected_data[timestamp] = text
                #print(f"Detected: {text}, Confidence: {confidence:.2f}")

                # Draw results on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{text} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            old_gray = frame_gray.copy()
            roi_points = new_points
        # if tracker update unsuccessful
        else:
            print("Tracking lost. Re-select ROI...")
            roi = select_roi(frame)
            x, y, w, h = roi
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            roi_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32).reshape(-1, 1, 2)
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # Initialize the tracker
            #tracker = cv2.TrackerKCF_create()
            # tracker = cv2.TrackerKCF_create()
            # tracker.init(frame, roi)
            print("ROI selected. Starting video processing...")
        # Display the current frame
        cv2.imshow("Video Processing", frame)
        frame_count += frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            print("Manual adjustment of ROI...")
            roi = select_roi(frame)
            # Initialize the tracker
            #tracker = cv2.TrackerKCF_create()
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, roi)
            print("ROI selected. Continuing video processing...")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_data


if __name__ == "__main__":
    #video_path = r"C:\Users\erutkovs\OneDrive - University College London\MRes sVNS project\Human trial\human_trial_recordings\data_06012025_pat_14\video\Human 014 060125\014_sVNS_C_1.6mA 1ms 20Hz 30s~3.mp4"  # Replace with the path to your video file
    #video_path = r"../../data_06012025_pat_14\video\Human 014 060125\014_sVNS_C_1.6mA 1ms 20Hz 30s~3.mp4"
    video_path = "../../data_06012025_pat_14/video/Human 014 060125/014_sVNS_P_900uA 1ms 20Hz 30s~2.mp4"
    detected_data = extract_text_from_video(video_path)
    #print("Final Detected Data:", detected_data)
    write_to_csv("../../data_06012025_pat_14/video/processed/014_sVNS_P_900uA 1ms 20Hz 30s~2.csv", detected_data)