import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    total_fingers = 0

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            lm_list = []
            fingers = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            label = handedness.classification[0].label

            # THUMB
            if label == "Right":
                if lm_list[4][0] > lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lm_list[4][0] < lm_list[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # OTHER FINGERS
            for i in range(1, 5):
                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers += sum(fingers)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(img, str(total_fingers), (50,150),
                cv2.FONT_HERSHEY_SIMPLEX, 4,
                (0,255,0), 8)

    cv2.imshow("Finger Counter (0-10)", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()