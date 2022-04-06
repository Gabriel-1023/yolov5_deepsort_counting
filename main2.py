#!/usr/bin/python
# -- coding:utf-8 --
"""
@Time ： 2022/4/4 20:35
@Auth ： Yan Zeyu
@File ： main2.py
@IDE ： PyCharm

"""
import time

import tracker
from detector import Detector
import cv2

if __name__ == '__main__':
    resize = (960, 540)
    list_bboxes = []
    counter = 0

    # 初始化 yolov5
    detector = Detector()
    # 打开视频
    # capture = cv2.VideoCapture('./video/test.mp4')
    capture = cv2.VideoCapture(0)
    start_time = time.time()
    while True:
        _, im = capture.read()
        if im is None:
            break
        im = cv2.resize(im, resize)
        bboxes = detector.detect(im)
        if len(bboxes) > 0:
            # before: [(123, 396, 179, 537, 'person', tensor(0.84004, device='cuda:0')), ...]
            list_bboxes = tracker.update(bboxes, im)
            # after: [(121, 396, 180, 536, 'person', 1), (399, 116, 432, 211, 'person', 2), ...]
            # 画框
            output_image_frame = tracker.draw_bboxes(im, list_bboxes, line_thickness=None)
        else:
            output_image_frame = im
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:  # 实时显示帧数
            cv2.putText(im, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # cv2.imshow('demo', output_image_frame)
        cv2.imshow('demo', im)
        cv2.waitKey(1)
    capture.release()
    cv2.destroyAllWindows()
