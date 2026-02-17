import cv2
import mediapipe
import numpy

hands_tool = mediapipe.solutions.hands
hands = hands_tool.Hands(max_num_hands=1, min_tracking_confidence=0.65)
drawing_fingers_tools = mediapipe.solutions.drawing_utils

camera_video = cv2.VideoCapture(0)

# red, yellow, green, pink
palette = [(250, 100, 100), (255, 250, 158), (171, 255, 158), (255, 158, 213)]
color_idx = 0
brush_thickness = 20

prev_finger_x, prev_finger_y = 0, 0

_, screen_frame = camera_video.read()
frame_h, frame_w, frame_c = screen_frame.shape
canvas = numpy.zeros((frame_h, frame_w, frame_c), dtype=numpy.uint8)

while True:
    success_read, frame_1 = camera_video.read()

    if not success_read:
        break

    frame = cv2.flip(frame_1, 1)

    height, width, channels = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_detected = hands.process(rgb)

    if hands_detected.multi_hand_landmarks:
        for hand_landmarks in hands_detected.multi_hand_landmarks:
            landmarks_positions = []

            for id, landmark in enumerate(hand_landmarks.landmark):
                landmarks_positions.append((int(landmark.x * width), int(landmark.y * height)))

            x_idx, y_idx = landmarks_positions[8]

            if prev_finger_x == 0 and prev_finger_y == 0:
                prev_finger_x, prev_finger_y = x_idx, y_idx

            idx_finger_dip_x, idx_finger_dip_y = landmarks_positions[7]

            if y_idx < idx_finger_dip_y:
                cv2.line(canvas, (prev_finger_x, prev_finger_y), (x_idx, y_idx), (34, 34, 189), brush_thickness)

            prev_finger_x, prev_finger_y = x_idx, y_idx

            drawing_fingers_tools.draw_landmarks(frame, hand_landmarks, hands_tool.HAND_CONNECTIONS)
    else:
        prev_finger_x, prev_finger_y = 0, 0

    frame = cv2.add(frame, canvas)
    cv2.imshow("Air Drawing App", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("You quit the program!")
        break

camera_video.release()
cv2.destroyAllWindows()

