import cv2
import math


class Face_detect:
    def __init__(self):
        cascade_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def face_detect(self, im):
        '''
        此函数进行人脸识别、展示图像以及返回人脸方位
        :param im: 图像ndarray
        :return: 0: 没有人脸, 1: 人脸在图像左边, 2: 人脸在图像右边
        '''
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        cv2.imshow('im', im)
        cv2.waitKey(10)
        return 0 if len(faces) == 0 else math.ceil(faces[0][0] / (im.shape[0] / 2))


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    test = Face_detect()
    while True:
        _, im = cam.read()
        print(test.face_detect(im))
