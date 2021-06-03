import  cv2 as cv

video=cv.VideoCapture(0)

faces=cv.CascadeClassifier('faces.xml')

while True:
    (ret,frame)=video.read()
    result = faces.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)
    for x,y,w,z in result:
        cv.rectangle(frame,(x,y),(x+w,y+z),(0,255,255),3)
        cv.rectangle(frame, (x, y), (x + 3, y), (0, 0, 255), 5)
        cv.rectangle(frame, (x, y), (x , y + 3), (0,0,255), 3)

    cv.imshow("frame",frame)
    frame=cv.flip(frame,1)


    k=cv.waitKey(1)
    if k==ord('q'):
        break
video.release()()
cv.destroyAllWindows()

