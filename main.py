import cv2
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('http://192.168.0.3:8080/video')
ret = True
i=0
while ret:
    ret, frame = cap.read()
    cv2.imshow('fr',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('circular_marker_shourya'+str(i)+'.jpg',frame)
        i = i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
