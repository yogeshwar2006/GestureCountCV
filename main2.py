import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

operation = "+"

while True:

    success, img = cap.read()
    img = cv2.flip(img,1)

    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    left_hand = 0
    right_hand = 0

    # Draw operation buttons
    cv2.rectangle(img,(w-120,50),(w-40,120),(255,0,0),-1)
    cv2.rectangle(img,(w-120,150),(w-40,220),(255,0,0),-1)
    cv2.rectangle(img,(w-120,250),(w-40,320),(255,0,0),-1)
    cv2.rectangle(img,(w-120,350),(w-40,420),(255,0,0),-1)

    cv2.putText(img,"+",(w-95,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(img,"-",(w-95,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(img,"*",(w-95,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(img,"/",(w-95,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            lm_list = []
            fingers = []

            for id, lm in enumerate(hand_landmarks.landmark):

                cx = int(lm.x*w)
                cy = int(lm.y*h)

                lm_list.append((cx,cy))

                # index finger tip for touch
                if id == 8:
                    cv2.circle(img,(cx,cy),10,(0,255,0),-1)

                    if w-120 < cx < w-40:

                        if 50 < cy < 120:
                            operation = "+"

                        elif 150 < cy < 220:
                            operation = "-"

                        elif 250 < cy < 320:
                            operation = "*"

                        elif 350 < cy < 420:
                            operation = "/"

            label = handedness.classification[0].label

            # Thumb
            if label == "Right":
                fingers.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
            else:
                fingers.append(1 if lm_list[4][0] < lm_list[3][0] else 0)

            # Other fingers
            for i in range(1,5):
                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            count = sum(fingers)

            if label == "Left":
                left_hand = count
            else:
                right_hand = count

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculation
    if operation == "+":
        result = left_hand + right_hand

    elif operation == "-":
        result = left_hand - right_hand

    elif operation == "*":
        result = left_hand * right_hand

    elif operation == "/":
        result = right_hand if right_hand == 0 else round(left_hand / right_hand,2)

    equation = f"{left_hand} {operation} {right_hand} = {result}"

    cv2.putText(img, equation, (20,60),
                cv2.FONT_HERSHEY_SIMPLEX,1.8,(0,255,0),4)

    cv2.imshow("Gesture Calculator", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()