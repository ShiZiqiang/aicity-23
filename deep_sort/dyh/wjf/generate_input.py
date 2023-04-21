"""
@Author: Du Yunhao
@Filename: generate_input.py
@Contact: dyh_bupt@163.com
@Time: 2021/8/23 9:28
@Discription: 基于跟踪结果，生成分类网络输入，用于获得分类结果
"""
import re
import os
import json
import time
import numpy as np
from os.path import join
from multiprocessing import Queue, Process

def run_2(q, j):
    while True:
        get = q.get()

        if isinstance(get, type(-1)):
            j.put(-1)
            break
        try:
            
            out = generate_input(
                dir_in_mot=r'/home/sugar/workspace/train_data',
                path_out_json=r'E:\Dataspace\Results\ActEV2021\mot\MOTv1SI_IoU±16_thres.9_and_traversal10_filtered.json',
                duration=32,
                stride=16,
                classid=-1,
                file=get
            )
            j.put(out)

        except Exception as e:
            print(e)
            continue



def monitor(q):
    while True:
        print('剩余:', q.qsize())
        if q.qsize() == 0:
            break
        time.sleep(5)

def save(j):
    out = []
    count = 0
    while True:
        get = j.get()
        if isinstance(get, type(-1)):
            count+=1
            if count == process_num:
                break
            continue
        out.extend(get)
    print(
        len(out)
    )
    # dump(
    #     out,
    #     open(r'/home/dyh/track.json', 'w'),
    #     indent=4,
    #     list_keys=['video', 'label', 'fid', 'duration', 'objType', 'bbox_clip', 'bbox_frame', 'trackID']
    # )
    json.dump(out, open(motTxt_out_dir, 'w'), indent=2)

'''Json格式化输出'''
def dump(obj, fp, indent=4, list_keys=None):
    str_indent = ' ' * indent
    iterable = list(json.JSONEncoder().iterencode(obj))
    for i, chunk in enumerate(iterable):
        if chunk == '{' and iterable[i+1] == '"video"':
            fp.write('\n' + str_indent + chunk)
        elif chunk.replace('"', '') in list_keys:
            fp.write('\n' + 2 * str_indent + chunk)
        elif iterable[i-1] == '"bbox_frame"':
            fp.write(chunk + '\n' + 3 * str_indent)
        elif re.match('"[0-9]+"', chunk):
            fp.write('\n' + 4 * str_indent + chunk)
        elif chunk == '}' and iterable[i-1] == '}' and iterable[i+1] == '}':
            fp.write('\n' + 3 * str_indent + chunk + '\n' + str_indent)
        else:
            fp.write(chunk)

def generate_input(dir_in_mot, path_out_json, duration, stride, classid, file):
    list_out = list()
    print('processing the file {}...'.format(file))
    array_in = np.loadtxt(join(dir_in_mot, file), delimiter=',')
    array_in = array_in[array_in[:, 7] == classid]
    array_in = array_in[
        np.lexsort([
            array_in[:, 0],
            array_in[:, 1]
        ])
    ]  # 先按ID排序，再按帧排序
    set_id = set(array_in[:, 1])
    video = file.replace('_mot.txt', '.avi')
    for id_ in set_id:
        array_id = array_in[array_in[:, 1] == id_]
        dict_frame2xyxy = {row[0]: [row[2], row[3], row[2] + row[4], row[3] + row[5]] for row in array_id}
        frame_start = int(array_id[0, 0])
        frame_stop = int(array_id[-1, 0])
        for f_start in range(frame_start, frame_stop - stride + 1, stride):
            f_stop = min(frame_stop, f_start + duration - 1)
            dict_bbox = dict()
            clip_bbox = [2000, 2000, -1, -1]
            for fid in range(f_start, f_stop + 1):
                bbox = dict_frame2xyxy[fid]
                dict_bbox[fid] = {
                    'big': bbox,
                    'small': bbox
                }
                clip_bbox[0] = min(clip_bbox[0], bbox[0])
                clip_bbox[1] = min(clip_bbox[1], bbox[1])
                clip_bbox[2] = max(clip_bbox[2], bbox[2])
                clip_bbox[3] = max(clip_bbox[3], bbox[3])
            list_out.append({
                'video': video,
                'fid': f_start,
                "label": {"person": 1},
                'duration': len(dict_bbox),
                'trackID': int(id_),
                'bbox_clip': clip_bbox,
                'bbox_frame': dict_bbox
            })
    return list_out
    '''保存'''
    dump(
        list_out,
        open(path_out_json, 'w'),
        indent=4,
        list_keys=['video', 'label', 'fid', 'duration', 'objType', 'bbox_clip', 'bbox_frame', 'trackID']
    )
    return None

if __name__ == '__main__':
    q = Queue()
    j = Queue()
    process_num = 60
    process_pool = []
    dir_in = r'/home/sugar/workspace/train_data'
    motTxt_out_dir = r'/home/dyh/track_train.json'
    
    for file in os.listdir(dir_in):
        if '_interpolate.txt' in file:
            q.put(file)

    for i in range(process_num):
        q.put(-1)
    for i in range(process_num):
        p = Process(target=run_2, args=(q, j))
        process_pool.append(p)

    p = Process(target=save, args=(j, ))
    process_pool.append(p)
    p = Process(target=monitor, args=(j, ))
    process_pool.append(p)
    for p in process_pool:
        p.start()
    for p in process_pool:
        p.join()
