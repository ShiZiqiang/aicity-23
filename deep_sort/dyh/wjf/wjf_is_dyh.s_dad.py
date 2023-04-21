import os
from multiprocessing import Queue, Process
import time
import glob
import sys
sys.path.append('../../')
from dyh.dyh_generate_detections import dyh_generate_detections_meva
from tools.generate_detections import create_box_encoder
from os.path import join, split

from deep_sort_app import *
from dyh.dyh_generate_detections import list_valVideo, list_testVideo

def run_1(q):
    model_filename = '/home/dyh/workspace/data1/dyh/models/DeepSORT/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=32)
    while True:
        get = q.get()

        if isinstance(get, type(-1)):
            break
        dir_in_frame, path_in_det, path_out_det = get
        try:
            dyh_generate_detections_meva(encoder, dir_in_frame, path_in_det, path_out_det)

        except Exception as e:
            print(e)
            continue

def run_2(q):
    while True:
        get = q.get()

        if isinstance(get, type(-1)):
            break
        x, y, z = get
        try:
            run(
                sequence_dir=x,
                detection_file=y,
                output_file=z,
                min_confidence=0.3,
                nms_max_overlap=1.0,
                min_detection_height=0,
                max_cosine_distance=0.4, # 0.5
                nn_budget=100,
                display=False,
            )

        except Exception as e:
            print(e)
            continue



def monitor(q):
    while True:
        print('剩余:', q.qsize())
        if q.qsize() == 0:
            break
        time.sleep(5)




# if __name__ == '__main__':
#     q = Queue()
#     process_num = 1
#     process_pool = []

#     print('loading encoder...')
    
    
#     dir_in_det = '/home/sugar/workspace/AICITY'
#     dir_out_det = dir_in_det
#     print('processing...')
#     root_frames = '/home/sugar/workspace/AICITY'
#     for i, file in enumerate(glob.glob(join(dir_in_det, '*0403.txt')), start=1):
#         print('  processing the {}th file {}...'.format(i, file))
#         if os.path.exists(file.replace('.txt', '.npy')):
#             continue
#         video = split(file)[1][:-4]
        
#         q.put((join(root_frames, 'testA_' + str(file[-10]) + '.mp4'), file, file.replace('txt', 'npy')))

#     for i in range(process_num):
#         q.put(-1)
#     for i in range(process_num):
#         p = Process(target=run_1, args=(q, ))
#         process_pool.append(p)
#     p = Process(target=monitor, args=(q, ))
#     process_pool.append(p)
#     for p in process_pool:
#         p.start()
#     for p in process_pool:
#         p.join()


if __name__ == '__main__':
    q = Queue()
    process_num = 1
    process_pool = []

    '''MEVA'''
    import glob
    from os.path import join, split, exists
    frame_root = '/home/sugar/workspace/AICITY'
    detections_dir = '/home/sugar/workspace/AICITY/cls_npy'
    output_dir = '/home/sugar/workspace/AICITY/cls_npy'
    for i, file in enumerate(glob.glob(join(detections_dir, '*0409.npy')), start=1):
        # if exists(file.replace('.npy', '_mot.txt')): continue
        print('processing the {}th file {}'.format(i, file))
        video = split(file)[1][:-4] + '.avi'
        q.put((join(frame_root, 'testA_' + str(file[-10]) + '.mp4'), file,file.replace('0409.npy', '0409_mot.txt')))

    for i in range(process_num):
        q.put(-1)
    for i in range(process_num):
        p = Process(target=run_2, args=(q, ))
        process_pool.append(p)
    p = Process(target=monitor, args=(q, ))
    process_pool.append(p)
    for p in process_pool:
        p.start()
    for p in process_pool:
        p.join()
    
    
        

