import os
import numpy as np
import sys
import time
import cv2
import json

np.set_printoptions(precision=5, threshold=np.inf, suppress=True)

if __name__ == '__main__':
    a = json.load(open('/data1/dyh/Results/ActEV2021/detection-v1_val54.bbox.json', 'r'))
    b = list(filter(lambda x: x['score'] >= 0.3, a))
    json.dump(b, open('/data1/dyh/Results/ActEV2021/detection-v1_val54_conf.3.bbox.json', 'w'))