import cv2
cap = cv2.VideoCapture('http://192.168.1.6:3000/video')
ret = True
while ret:
    ret, frame = cap.read()
    cv2.imshow('fr',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
