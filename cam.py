import cv2 as cv
cam=cv.VideoCapture(0)
cv.namedWindow("web Cam apply")
img_counter=0
def mi(fram):
    img=cv.flip(fram,1)
    return img

def Col(fram):
    img=cv.cvtColor(fram,cv.COLOR_BGR2GRAY)
    return img

def size(fram,scale=.7):
    height=int(fram.shape[1]*scale)
    width=int(fram.shape[1]*scale)
    dimension=(height,width)
    return cv.resize(fram,dimension,interpolation=cv.INTER_AREA)


def face(fram):

    reader=cv.CascadeClassifier("faces.xml")
    result=reader.detectMultiScale(fram,scaleFactor=1.1,minNeighbors=3)
    return result


while True:
    ret,frame=cam.read()
    frame=Col(frame)
    frame=size(frame)
    result=face(frame)
    frame=mi(frame)

    for x,y,w,z in result:
        cv.rectangle(frame,(x,y),(x+w,y+z),(0,255,234),thickness=3)
        cv.putText(frame, 'face num' + str(frame.shape[0]), (5, 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cam.release()
cv.destroyAllWindows()
