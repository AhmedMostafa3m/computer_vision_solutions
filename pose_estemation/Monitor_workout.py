import cv2
import ultralytics
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download

#safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/Pushups.demo.video.mp4")

cap = cv2.VideoCapture("07-pose_estemation\\Pushups.demo.video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("Pushups.demo.video.output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init AIGym
gym = solutions.AIGym(
    show=True,  # Display the frame
    kpts=[5, 7, 9],  # keypoints index of person for monitoring specific exercise, by default it's for pushup
    model="yolo11n-pose.pt",  # Path to the YOLO11 pose estimation model file
    pose_type="pushup",  # Type of exercise to monitor
    line_width=4,  # Adjust the line width for bounding boxes and text display
    verbose=False,
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        
        im0 = gym.monitor(im0)  # monitor workouts on each frame
        video_writer.write(im0)  # write the output frame in file.

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()