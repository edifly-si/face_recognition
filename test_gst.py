import cv2

gst = (
    "rtspsrc location=rtsp://admin:S3mangat45%2A%2A@192.168.1.65 latency=50 ! "
    "rtph265depay ! h265parse ! "
    "nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
print("Opened:", cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break

    cv2.imshow("RTSP", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
