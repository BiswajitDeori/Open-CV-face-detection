import cv2 as cv
faces=cv.CascadeClassifier('faces.xml')
class Video(object):
   def __init__(self):
      self.video=cv.VideoCapture(0)
   def __del__(self):
       self.video.release()
   def get_frame(self):
      (ret, frame)=self.video.read()
      frame = cv.flip(frame,1)
      result = faces.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2)
      for x, y, w, z in result:
         cv.rectangle(frame, (x, y), (x + w, y + z), (0, 255, 255), 3)
         cv.rectangle(frame, (x, y), (x + 3, y), (0, 0, 255), 5)
         cv.rectangle(frame, (x, y), (x, y + 3), (0, 0, 255), 3)
         cv.putText(frame,"NO OF FACES "+str(len(result)), (40,40), cv.FONT_ITALIC, 0.8, (0,255,112), 2)
      (ret, jpg)=cv.imencode('.jpg',frame)
      return jpg.tobytes()