import cv2
import time
import csv

# Path to your video file
video_path = "../data_06012025_pat_14/video/Human 014 060125/014_sVNS_P_900uA 1ms 20Hz 30s.mp4"
cap = cv2.VideoCapture(video_path)

# Start time for logging relative events
start_time = time.time()
events = []  # List to store events as dictionaries
cv2.namedWindow("RR verification", cv2.WINDOW_KEEPRATIO)
print("Press 'i' for inhale, 'e' for exhale. Press 'q' to quit.")

ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(500 / fps)
# Rotate if there's an issue with the smartphone metadata interpretation
# frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

# Display the first frame
cv2.imshow("RR verification", frame)
print("\nReady?")
    
while cap.isOpened():
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('y'):
        while cap.isOpened():
            ret, frame = cap.read() # read the next frame
            if not ret:
                break
            # Rotate if there's an issue with the smartphone metadata interpretation
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cv2.putText(frame, f"Time elapsed: {timestamp: .2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow("RR verification", frame)
            key2 = cv2.waitKey(delay) & 0xFF
            if key2 == ord('q'):
                break
            elif key2 == ord('i'):
                #event_time = time.time() - start_time
                event_time = timestamp
                events.append({"event": "inhale", "time": event_time})
                print(f"Inhale at {event_time:.2f} seconds")
            elif key2 == ord('e'):
                #event_time = time.time() - start_time
                event_time = timestamp
                events.append({"event": "exhale", "time": event_time})
                print(f"Exhale at {event_time:.2f} seconds")
        
        cap.release()
        cv2.destroyAllWindows()
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
# Save the events to a CSV file for later analysis
csv_path = "../data_06012025_pat_14/video/processed/014_RR_verification_test.csv"
with open(csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['event', 'time']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for entry in events:
        writer.writerow(entry)

print(f"Event log saved to {csv_path}")
