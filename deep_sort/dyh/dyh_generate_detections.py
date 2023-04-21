import os
from os.path import join, split
import json
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
from pprint import pprint
import sys
sys.path.append('../')
from tools.generate_detections import create_box_encoder

list_valVideo = [
                'VIRAT_S_000007.mp4',
                'VIRAT_S_000008.mp4',
                'VIRAT_S_000200_00_000100_000171.mp4',
                'VIRAT_S_000200_02_000479_000635.mp4',
                'VIRAT_S_000201_00_000018_000380.mp4',
                'VIRAT_S_000201_01_000384_000589.mp4',
                'VIRAT_S_000201_02_000590_000623.mp4',
                'VIRAT_S_000201_04_000682_000822.mp4',
                'VIRAT_S_000203_01_000171_000345.mp4',
                'VIRAT_S_000203_08_001702_001734.mp4',
                'VIRAT_S_000204_07_001577_001611.mp4',
                'VIRAT_S_000204_09_001768_001849.mp4',
                # 'VIRAT_S_000205_01_000197_000342.mp4',  #
                'VIRAT_S_000205_02_000409_000566.mp4',
                'VIRAT_S_000205_03_000860_000922.mp4',
                'VIRAT_S_000205_05_001092_001124.mp4',
                'VIRAT_S_000205_06_001566_001631.mp4',
                'VIRAT_S_000206_00_000025_000058.mp4',
                'VIRAT_S_000206_01_000148_000184.mp4',
                'VIRAT_S_000206_02_000294_000327.mp4',
                'VIRAT_S_000206_08_001618_001712.mp4',
                'VIRAT_S_000207_02_000498_000530.mp4',
                'VIRAT_S_000207_03_000556_000590.mp4',
                'VIRAT_S_040000_00_000000_000036.mp4',
                'VIRAT_S_040000_01_000042_000099.mp4',
                'VIRAT_S_040000_04_000532_000622.mp4',
                'VIRAT_S_040000_08_001084_001190.mp4',
                'VIRAT_S_040000_09_001194_001574.mp4',
                'VIRAT_S_040003_00_000000_000072.mp4',
                'VIRAT_S_040003_01_000083_000193.mp4',
                'VIRAT_S_040003_02_000197_000552.mp4',
                'VIRAT_S_040005_06_000886_001016.mp4',
                'VIRAT_S_040005_08_001225_001276.mp4',
                'VIRAT_S_040100_03_000496_000559.mp4',
                'VIRAT_S_040100_04_000626_000689.mp4',
                'VIRAT_S_040100_05_000696_000762.mp4',
                'VIRAT_S_040100_06_000767_000988.mp4',
                'VIRAT_S_040100_07_001043_001099.mp4',
                'VIRAT_S_040100_08_001103_001181.mp4',
                'VIRAT_S_040100_09_001186_001533.mp4',
                'VIRAT_S_040101_05_000722_001547.mp4',
                'VIRAT_S_040101_06_001557_001590.mp4',
                'VIRAT_S_040102_04_000596_000689.mp4',
                'VIRAT_S_040102_06_000849_000908.mp4',
                'VIRAT_S_040102_07_000916_000983.mp4',
                'VIRAT_S_040103_00_000000_000120.mp4',
                'VIRAT_S_040103_01_000132_000195.mp4',
                'VIRAT_S_040103_02_000199_000279.mp4',
                'VIRAT_S_040103_03_000284_000425.mp4',
                'VIRAT_S_040103_06_000836_000909.mp4',
                'VIRAT_S_040104_07_001268_001348.mp4',
                'VIRAT_S_050000_06_000908_000970.mp4',
                'VIRAT_S_050000_08_001235_001295.mp4',
                'VIRAT_S_050000_09_001310_001373.mp4',
                'VIRAT_S_050000_11_001530_001576.mp4']

list_testVideo = ['VIRAT_S_000003.mp4',
                               'VIRAT_S_000100.mp4',
                               'VIRAT_S_000103.mp4',
                               'VIRAT_S_000104.mp4',
                               'VIRAT_S_000106.mp4',
                               'VIRAT_S_000200_04_000937_001443.mp4',
                               'VIRAT_S_000200_06_001693_001824.mp4',
                               'VIRAT_S_000202_01_001334_001520.mp4',
                               'VIRAT_S_000202_05_001768_001849.mp4',
                               'VIRAT_S_000203_00_000128_000160.mp4',
                               'VIRAT_S_000203_02_000357_000443.mp4',
                               'VIRAT_S_000204_01_000119_000219.mp4',
                               'VIRAT_S_000204_06_001527_001560.mp4',
                               'VIRAT_S_000204_08_001704_001746.mp4',
                               'VIRAT_S_000206_05_001231_001329.mp4',
                               'VIRAT_S_000206_06_001421_001458.mp4',
                               'VIRAT_S_000207_00_000000_000045.mp4',
                               'VIRAT_S_000207_05_001125_001193.mp4',
                               'VIRAT_S_000300_00_000022_000054.mp4',
                               'VIRAT_S_000300_01_000055_000218.mp4',
                               'VIRAT_S_000300_02_000355_000398.mp4',
                               'VIRAT_S_000300_03_000406_000449.mp4',
                               'VIRAT_S_000300_04_000491_000570.mp4',
                               'VIRAT_S_000300_05_000579_000715.mp4',
                               'VIRAT_S_000300_09_001223_001264.mp4',
                               'VIRAT_S_000300_10_001323_001365.mp4',
                               'VIRAT_S_000300_11_001391_001439.mp4',
                               'VIRAT_S_000300_12_001451_001500.mp4',
                               'VIRAT_S_010000_02_000388_000421.mp4',
                               'VIRAT_S_010000_03_000442_000528.mp4',
                               'VIRAT_S_010000_05_000638_000718.mp4',
                               'VIRAT_S_010000_07_000827_000860.mp4',
                               'VIRAT_S_010000_08_000893_001024.mp4',
                               'VIRAT_S_010001_00_000019_000065.mp4',
                               'VIRAT_S_010001_02_000195_000498.mp4',
                               'VIRAT_S_010001_03_000537_000563.mp4',
                               'VIRAT_S_010001_04_000583_000646.mp4',
                               'VIRAT_S_010001_06_000685_000722.mp4',
                               'VIRAT_S_010001_07_000741_000810.mp4',
                               'VIRAT_S_010001_08_000826_000893.mp4',
                               'VIRAT_S_010001_09_000921_000952.mp4',
                               'VIRAT_S_010001_10_000962_001005.mp4',
                               'VIRAT_S_010002_00_000012_000120.mp4',
                               'VIRAT_S_010002_01_000123_000148.mp4',
                               'VIRAT_S_010002_02_000174_000204.mp4',
                               'VIRAT_S_010002_05_000397_000420.mp4',
                               'VIRAT_S_010002_06_000441_000467.mp4',
                               'VIRAT_S_010003_03_000219_000259.mp4',
                               'VIRAT_S_010003_08_000739_000778.mp4',
                               'VIRAT_S_010108_00_000000_000555.mp4',
                               'VIRAT_S_010109_01_000065_000183.mp4',
                               'VIRAT_S_010109_02_000208_000318.mp4',
                               'VIRAT_S_010109_03_000321_000473.mp4',
                               'VIRAT_S_010109_06_000725_000849.mp4',
                               'VIRAT_S_010110_01_000024_000240.mp4',
                               'VIRAT_S_010111_01_000092_000156.mp4',
                               'VIRAT_S_010111_03_000324_000608.mp4',
                               'VIRAT_S_010112_02_000221_000372.mp4',
                               'VIRAT_S_010112_03_000400_000726.mp4',
                               'VIRAT_S_010112_04_000728_000916.mp4',
                               'VIRAT_S_010113_01_000122_000418.mp4',
                               'VIRAT_S_010113_03_000505_000639.mp4',
                               'VIRAT_S_010115_00_000064_000383.mp4',
                               'VIRAT_S_010116_01_000182_000221.mp4',
                               'VIRAT_S_010116_02_000236_000284.mp4',
                               'VIRAT_S_010116_03_000628_000672.mp4',
                               'VIRAT_S_010116_04_000850_000887.mp4',
                               'VIRAT_S_010116_05_000895_000927.mp4',
                               'VIRAT_S_010116_06_000936_000969.mp4',
                               'VIRAT_S_010117_00_000086_000155.mp4',
                               'VIRAT_S_010117_01_000267_000322.mp4',
                               'VIRAT_S_010117_02_000363_000395.mp4',
                               'VIRAT_S_010117_03_000609_000646.mp4',
                               'VIRAT_S_010117_04_000844_000880.mp4',
                               'VIRAT_S_010117_05_000893_000937.mp4',
                               'VIRAT_S_010117_06_000969_000999.mp4',
                               'VIRAT_S_040000_00_000063_000085.mp4',
                               'VIRAT_S_040000_06_000746_000964.mp4',
                               'VIRAT_S_040000_07_000966_001071.mp4',
                               'VIRAT_S_040001_02_001102_001530.mp4',
                               'VIRAT_S_040002_00_000000_000377.mp4',
                               'VIRAT_S_040002_01_000380_000718.mp4',
                               'VIRAT_S_040002_02_000720_001405.mp4',
                               'VIRAT_S_040003_03_000577_000741.mp4',
                               'VIRAT_S_040003_05_001123_001437.mp4',
                               'VIRAT_S_040004_01_000117_000238.mp4',
                               'VIRAT_S_040004_02_000239_000301.mp4',
                               'VIRAT_S_040004_03_000321_000516.mp4',
                               'VIRAT_S_040004_04_000549_000643.mp4',
                               'VIRAT_S_040004_05_000655_000738.mp4',
                               'VIRAT_S_040004_06_000770_000852.mp4',
                               'VIRAT_S_040004_07_000860_000929.mp4',
                               'VIRAT_S_040004_08_000949_001057.mp4',
                               'VIRAT_S_040004_09_001079_001176.mp4',
                               'VIRAT_S_040004_10_001180_001382.mp4',
                               'VIRAT_S_040004_11_001397_001482.mp4',
                               'VIRAT_S_040004_12_001504_001581.mp4',
                               'VIRAT_S_040005_01_000118_000185.mp4',
                               'VIRAT_S_040005_02_000190_000260.mp4',
                               'VIRAT_S_040005_03_000275_000345.mp4',
                               'VIRAT_S_040005_04_000354_000579.mp4',
                               'VIRAT_S_040005_05_000581_000884.mp4',
                               'VIRAT_S_040005_09_001323_001440.mp4',
                               'VIRAT_S_040005_10_001453_001515.mp4',
                               'VIRAT_S_040005_11_001530_001595.mp4',
                               'VIRAT_S_040006_00_000000_000080.mp4',
                               'VIRAT_S_040006_01_000103_000459.mp4',
                               'VIRAT_S_040006_02_000462_000659.mp4',
                               'VIRAT_S_040006_03_000714_000756.mp4',
                               'VIRAT_S_040100_02_000435_000490.mp4',
                               'VIRAT_S_040101_00_000000_000088.mp4',
                               'VIRAT_S_040101_01_000093_000197.mp4',
                               'VIRAT_S_040102_00_000000_000269.mp4',
                               'VIRAT_S_040102_01_000275_000362.mp4',
                               'VIRAT_S_040102_05_000692_000756.mp4',
                               'VIRAT_S_040102_09_001104_001253.mp4',
                               'VIRAT_S_040102_11_001411_001473.mp4',
                               'VIRAT_S_040102_12_001476_001552.mp4',
                               'VIRAT_S_040103_04_000432_000726.mp4',
                               'VIRAT_S_040104_02_000459_000721.mp4',
                               'VIRAT_S_040104_03_000726_000851.mp4',
                               'VIRAT_S_040104_09_001475_001583.mp4',
                               'VIRAT_S_040105_00_000007_000135.mp4',
                               'VIRAT_S_040105_01_000202_000449.mp4',
                               'VIRAT_S_040105_02_000472_000599.mp4',
                               'VIRAT_S_040105_03_000607_000664.mp4',
                               'VIRAT_S_040105_04_000686_000797.mp4',
                               'VIRAT_S_040105_05_000804_000859.mp4',
                               'VIRAT_S_040105_06_000868_001041.mp4',
                               'VIRAT_S_040105_07_001063_001458.mp4',
                               'VIRAT_S_040105_08_001463_001590.mp4',
                               'VIRAT_S_040106_00_000000_000012.mp4',
                               'VIRAT_S_040106_01_000018_000138.mp4',
                               'VIRAT_S_040106_02_000142_000200.mp4',
                               'VIRAT_S_040106_03_000215_000292.mp4',
                               'VIRAT_S_040106_04_000297_000712.mp4',
                               'VIRAT_S_040106_05_000724_000864.mp4',
                               'VIRAT_S_040106_06_000869_000996.mp4',
                               'VIRAT_S_040106_07_000999_001043.mp4',
                               'VIRAT_S_040106_08_001046_001271.mp4',
                               'VIRAT_S_040201_00_000152_000456.mp4',
                               'VIRAT_S_040201_01_000741_001112.mp4',
                               'VIRAT_S_040201_02_001252_001297.mp4',
                               'VIRAT_S_040201_03_001339_001361.mp4',
                               'VIRAT_S_040201_04_001418_001454.mp4',
                               'VIRAT_S_040201_05_001500_001554.mp4',
                               'VIRAT_S_040202_00_000023_000069.mp4',
                               'VIRAT_S_040202_01_000104_000132.mp4',
                               'VIRAT_S_040202_02_000177_000212.mp4',
                               'VIRAT_S_040202_03_000292_000406.mp4',
                               'VIRAT_S_040202_04_000732_001276.mp4',
                               'VIRAT_S_040202_05_001415_001474.mp4',
                               'VIRAT_S_040202_06_001529_001578.mp4',
                               'VIRAT_S_040203_00_000113_000339.mp4',
                               'VIRAT_S_040203_01_000427_000487.mp4',
                               'VIRAT_S_040203_02_000580_000628.mp4',
                               'VIRAT_S_040203_03_000938_001490.mp4',
                               'VIRAT_S_040204_01_000785_000794.mp4',
                               'VIRAT_S_040204_02_000834_000849.mp4',
                               'VIRAT_S_040204_03_000877_000884.mp4',
                               'VIRAT_S_040204_04_000917_000924.mp4',
                               'VIRAT_S_040204_05_000957_000994.mp4',
                               'VIRAT_S_040204_06_001028_001037.mp4',
                               'VIRAT_S_040204_07_001093_001133.mp4',
                               'VIRAT_S_040204_08_001176_001182.mp4',
                               'VIRAT_S_040204_09_001221_001281.mp4',
                               'VIRAT_S_040204_10_001319_001328.mp4',
                               'VIRAT_S_040204_11_001345_001354.mp4',
                               'VIRAT_S_040204_12_001372_001381.mp4',
                               'VIRAT_S_040204_13_001404_001413.mp4',
                               'VIRAT_S_040204_14_001469_001568.mp4',
                               'VIRAT_S_040205_00_000215_000645.mp4',
                               'VIRAT_S_040205_01_000732_000738.mp4',
                               'VIRAT_S_040205_02_000759_000769.mp4',
                               'VIRAT_S_040205_03_001018_001503.mp4',
                               'VIRAT_S_040206_00_000135_000405.mp4',
                               'VIRAT_S_040206_01_000755_001211.mp4',
                               'VIRAT_S_040207_00_000107_000322.mp4',
                               'VIRAT_S_040207_01_000443_000569.mp4',
                               'VIRAT_S_040207_02_000895_001468.mp4',
                               'VIRAT_S_040208_00_000092_000276.mp4',
                               'VIRAT_S_040208_01_000726_001461.mp4',
                               'VIRAT_S_040209_00_000475_001426.mp4',
                               'VIRAT_S_040210_00_000475_001426.mp4',
                               'VIRAT_S_040211_00_000475_001426.mp4',
                               'VIRAT_S_040212_00_000476_001428.mp4',
                               'VIRAT_S_040213_00_000009_000027.mp4',
                               'VIRAT_S_040303_01_001051_001119.mp4',
                               'VIRAT_S_040304_01_001296_001375.mp4',
                               'VIRAT_S_040401_00_000000_000174.mp4',
                               'VIRAT_S_040403_00_000799_000966.mp4',
                               'VIRAT_S_040404_00_000123_000388.mp4',
                               'VIRAT_S_040404_01_000497_000578.mp4',
                               'VIRAT_S_040404_03_000876_000904.mp4',
                               'VIRAT_S_040404_04_001090_001584.mp4',
                               'VIRAT_S_040405_00_000000_000137.mp4',
                               'VIRAT_S_040405_02_001126_001585.mp4',
                               'VIRAT_S_050000_02_000404_000486.mp4',
                               'VIRAT_S_050000_03_000585_000639.mp4',
                               'VIRAT_S_050000_13_001722_001766.mp4',
                               'VIRAT_S_050000_16_001947_001989.mp4',
                               'VIRAT_S_050100_00_000025_000075.mp4',
                               'VIRAT_S_050100_01_000095_000248.mp4',
                               'VIRAT_S_050100_02_000275_000376.mp4',
                               'VIRAT_S_050100_03_000380_000490.mp4',
                               'VIRAT_S_050100_04_000492_000606.mp4',
                               'VIRAT_S_050100_06_000845_000915.mp4',
                               'VIRAT_S_050100_07_000968_001080.mp4',
                               'VIRAT_S_050100_09_001199_001390.mp4',
                               'VIRAT_S_050100_12_002133_002194.mp4',
                               'VIRAT_S_050100_13_002202_002244.mp4',
                               'VIRAT_S_050101_01_000200_000353.mp4',
                               'VIRAT_S_050101_02_000400_000470.mp4',
                               'VIRAT_S_050101_06_000868_000929.mp4',
                               'VIRAT_S_050101_09_001427_001474.mp4',
                               'VIRAT_S_050101_10_001574_001735.mp4',
                               'VIRAT_S_050101_11_001850_002110.mp4',
                               'VIRAT_S_050102_01_000322_000406.mp4',
                               'VIRAT_S_050200_00_000106_000380.mp4',
                               'VIRAT_S_050201_00_000012_000116.mp4',
                               'VIRAT_S_050201_02_000395_000483.mp4',
                               'VIRAT_S_050201_03_000573_000647.mp4',
                               'VIRAT_S_050201_04_000669_000728.mp4',
                               'VIRAT_S_050201_05_000890_000944.mp4',
                               'VIRAT_S_050201_06_001168_001240.mp4',
                               'VIRAT_S_050201_08_001567_001647.mp4',
                               'VIRAT_S_050201_09_001821_001876.mp4',
                               'VIRAT_S_050201_10_001992_002056.mp4',
                               'VIRAT_S_050201_11_002090_002220.mp4',
                               'VIRAT_S_050202_01_000150_000280.mp4',
                               'VIRAT_S_050202_03_000608_000684.mp4',
                               'VIRAT_S_050202_04_000690_000750.mp4',
                               'VIRAT_S_050202_05_000857_000912.mp4',
                               'VIRAT_S_050202_06_001048_001115.mp4',
                               'VIRAT_S_050202_07_001126_001301.mp4',
                               'VIRAT_S_050202_08_001410_001494.mp4',
                               'VIRAT_S_050202_09_001642_001712.mp4',
                               'VIRAT_S_050202_10_002159_002233.mp4',
                               'VIRAT_S_050203_00_000023_000097.mp4',
                               'VIRAT_S_050203_01_000147_000223.mp4',
                               'VIRAT_S_050203_02_000282_000346.mp4',
                               'VIRAT_S_050203_03_000441_000534.mp4',
                               'VIRAT_S_050203_05_000980_001061.mp4',
                               'VIRAT_S_050203_06_001202_001264.mp4',
                               'VIRAT_S_050203_08_001686_001885.mp4',
                               'VIRAT_S_050203_10_002093_002215.mp4']

def dyh_generate_detections_1(encoder, frames_dir, det_json, output_path):
    """
    :param encoder: 特征编码器
    :param frames_dir: 视频帧目录，为帧图像的直接父目录
    :param det_json: 目标检测生成的COCO格式的json文件
    :param output_path: 生成的检测框与特征.npy存储路径
    Note: 保存格式为10维det信息+128维特征
        det信息：frame_id(从1开始), -1, x, y, w, h, score, -1, class_id, -1
    """
    video_name = os.path.split(frames_dir)[1]
    detections_out = list()
    '''process'''
    for i, dict_bbox in enumerate(det_json, start=1):
        '''信息提取与筛选'''
        frame_id, class_id, xywh, score = int(dict_bbox['image_id'] % pow(10, 5)), dict_bbox['category_id'], \
                                          np.clip(dict_bbox['bbox'], 0, 1920), dict_bbox['score']
        if score < 0.3: continue
        img = cv2.imread(os.path.join(frames_dir, 'frame_%05d.jpg'%frame_id), cv2.IMREAD_COLOR)
        try:
            features = encoder(img, [xywh])
            detections_out.append(
                [frame_id, -1, xywh[0], xywh[1], xywh[2], xywh[3], score, -1, class_id, -1] + features.tolist()[0]
            )
        except BaseException:
            print('dyhError: error occurs when processing {} frame {}...'.format(video_name, frame_id))
            f = open(path_error_log, 'a')
            f.write('dyhError: error occurs when processing {} frame {}...\n'.format(video_name, frame_id))
            f.close()
    '''save'''
    # print(np.array(detections_out).shape)
    np.save(output_path, np.array(detections_out), allow_pickle=False)

def dyh_generate_detections_2(encoder, dir_in_frame, path_in_det, path_out_det):
    """
    基于mot格式的检测结果生成对应特征文件，注意检测结果的格式！
    :param encoder: 编码器
    :param dir_in_frame: 帧目录
    :param path_in_det: 检测结果
    :param path_out_det: 特征文件
    """
    list_detections = list()
    matrix = np.loadtxt(path_in_det, delimiter=',', dtype=int)
    for det in matrix:
        frame_id, obj_id, x, y, w, h, score, class_, visibility = det
        if frame_id % 8 == 0:  # 隔8帧取1帧
            img = cv2.imread(
                join(
                    dir_in_frame, 'frame_%05d.jpg' % (frame_id + 1)
                )
            )
            features = encoder(img, [[x, y, w, h]])
            list_detections.append(
                [frame_id, -1, x, y, w, h, score, class_, visibility, -1] + features.tolist()[0]
            )
    np.save(path_out_det, np.array(list_detections), allow_pickle=False)

def dyh_generate_detections_meva(encoder, dir_in_frame, path_in_det, path_out_det):
    thres_score = 0.3
    list_detections = list()
    matrix = np.loadtxt(path_in_det, delimiter=',')
    for det in matrix:
        frame_id, obj_id, x, y, w, h, score, class_, visibility = det
        if score < thres_score: continue
        img = cv2.imread(join(dir_in_frame, 'frame_%05d.jpg' % frame_id))
        features = encoder(img, [[x, y, w, h]])
        list_detections.append(
            [frame_id, -1, x, y, w, h, score, -1, -1, visibility] + features.tolist()[0]
        )
    np.save(path_out_det, np.array(list_detections), allow_pickle=False)


if __name__ == '__main__':
    '''VIRAT'''
    # print('loading encoder...')
    # model_filename = '/data1/dyh/models/DeepSORT/mars-small128.pb'
    # encoder = create_box_encoder(model_filename, batch_size=32)
    # print('loading vehicle_det_json...')
    # det_json_path = '/data1/dyh/Results/ActEV2021/TEST/DET/DetectionFinal_person_vehicle_conf.5.bbox.json'
    # det_json = json.load(open(det_json_path, 'r'))
    # print('processing...')
    # frames_root = '/home/spring/virat_ActEV/virat_frame'
    # output_dir = '/data1/dyh/Results/ActEV2021/TEST/DET/DetectionFinal_features'
    # list_video = list_testVideo
    # for i, dir_ in enumerate(list_video, start=1):
    #     print('  {}/{} processing {}...'.format(i ,len(list_video), dir_))
    #     one_det_json = list(filter(lambda x: x['image_id'] // pow(10,5) == i, det_json))
    #     frames_dir = os.path.join(frames_root, dir_)
    #     output_path = os.path.join(output_dir, dir_.replace('.mp4', '.npy'))
    #     dyh_generate_detections_1(
    #         encoder=encoder,
    #         frames_dir=frames_dir,
    #         det_json=one_det_json,
    #         output_path=output_path
    #     )
    '''MEVA'''
    import glob
    print('loading encoder...')
    model_filename = '/home/dyh/workspace/data1/dyh/models/DeepSORT/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=32)
    dir_in_det = '/home/sugar/workspace/train_data'
    dir_out_det = dir_in_det
    print('processing...')
    root_frames = '/disk2/meva_data/meva_frames'
    for i, file in enumerate(glob.glob(join(dir_in_det, '*.txt')), start=1):
        print('  processing the {}th file {}...'.format(i, file))
        video = split(file)[1][:-4]
        dyh_generate_detections_meva(
            encoder=encoder,
            dir_in_frame=join(root_frames, video + '.avi'),
            path_in_det=file,
            path_out_det=file.replace('.txt', '.npy')
        )
