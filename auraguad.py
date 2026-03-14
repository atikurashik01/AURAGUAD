import cv2
import time
from ultralytics import YOLO

# --------------------------
# Load YOLO model
# --------------------------
model = YOLO("yolov8n.pt")

# --------------------------
# Webcam
# --------------------------
cap = cv2.VideoCapture(0)

# --------------------------
# COCO Class IDs
# --------------------------
PERSON = 0
BOTTLE = 39
CUP = 41
PHONE = 67

# --------------------------
# Timers
# --------------------------
person_last_seen = time.time()
phone_start = None
last_drink_seen = time.time()

# thresholds
# thresholds
EMPTY_THRESHOLD = 5
PHONE_THRESHOLD = 2
HYDRATION_THRESHOLD = 1800   # 30 minutes

print("AuraGuard running... Press Q to exit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    person_detected = False
    phone_detected = False
    drink_detected = False

    # --------------------------
    # Process detections
    # --------------------------
    for box in results.boxes:

        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if class_id == PERSON:
            person_detected = True
            person_last_seen = time.time()

        if class_id == PHONE:
            phone_detected = True

        if class_id in [BOTTLE, CUP]:
            drink_detected = True
            last_drink_seen = time.time()

        # draw detection box
        if class_id in [PERSON, PHONE, BOTTLE, CUP]:

            color = (0,255,0)
            label = model.names[class_id]

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,color,2)

    current_time = time.time()

    # --------------------------
    # STATE 1 : EMPTY DESK
    # --------------------------
    if not person_detected and (current_time - person_last_seen > EMPTY_THRESHOLD):

        cv2.putText(frame,
                    "System Paused: User Away",
                    (40,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)

    else:

        # --------------------------
        # Phone timer
        # --------------------------
        if phone_detected:

            if phone_start is None:
                phone_start = current_time

        else:
            phone_start = None

        phone_time = 0
        if phone_start:
            phone_time = current_time - phone_start

        hydration_time = current_time - last_drink_seen

        # --------------------------
        # STATE 3 : DISTRACTED
        # --------------------------
        if phone_detected and phone_time > PHONE_THRESHOLD:

            cv2.putText(frame,
                        "WARNING: PUT PHONE AWAY",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

        # --------------------------
        # STATE 4 : DEHYDRATED
        # --------------------------
        elif hydration_time > HYDRATION_THRESHOLD:

            cv2.putText(frame,
                        "HEALTH ALERT: Drink Water",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        3)

        # --------------------------
        # STATE 2 : DEEP WORK
        # --------------------------
        elif person_detected and not phone_detected:

            cv2.putText(frame,
                        "Status: Focusing",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        3)

    cv2.imshow("AuraGuard Live Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()