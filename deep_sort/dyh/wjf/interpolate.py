"""
@Author: Du Yunhao
@Filename: interpolate.py
@Contact: dyh_bupt@163.com
@Time: 2021/8/23 15:05
@Discription: 轨迹线性插值
"""
import os
import time
import numpy as np
from os.path import join, exists
from multiprocessing import Queue, Process

def run_2(q):
    while True:
        get = q.get()

        if isinstance(get, type(-1)):
            break
        try:
            
            interpotate(
                dir_in = r'/home/sugar/workspace/train_data',
                dir_out = r'/home/sugar/workspace/train_data',
                f=get
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

def interpotate(dir_in, dir_out, f):
    array_in = np.loadtxt(join(dir_in, f), delimiter=',')
    array_in = array_in[
        np.lexsort([
            array_in[:, 0],
            array_in[:, 1]
        ])
    ]  # 按ID和帧排序
    array_out = array_in.copy()
    '''线性插值'''
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in array_in:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:  # 同ID
            if f_pre + 1 < f_curr < f_pre + 1e5:
                for j, frame in enumerate(range(f_pre + 1, f_curr), start=1):  # 逐帧插值
                    '''框'''
                    bbox_pre = row_pre[2:6]
                    bbox_curr = row[2:6]
                    '''计算插值步幅'''
                    step_bbox = (bbox_curr - bbox_pre) / (f_curr - f_pre) * j
                    step_ohter = (row[6:] - row_pre[6:]) / (f_curr - f_pre) * j
                    '''插值'''
                    bbox_new = (bbox_pre + step_bbox).tolist()
                    other_new = (row_pre[6:] + step_ohter).tolist()
                    '''保存'''
                    array_out = np.append(
                        array_out,
                        np.array([[frame, id_curr] + bbox_new + other_new]),
                        axis=0
                    )
        else:  # 不同ID
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    '''保存'''
    array_out = array_out[
        np.lexsort([
            array_out[:, 0],
            array_out[:, 1]
        ])
    ]  # 按ID和帧排序
    np.savetxt(join(dir_out, f).replace('_smooth.txt', '_interpolate.txt'), array_out, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')

if __name__ == '__main__':
    q = Queue()
    process_num = 60
    process_pool = []
    dir_in = r'/home/sugar/workspace/train_data'
    motTxt_out_dir = r'/home/sugar/workspace/train_data'
    if not exists(motTxt_out_dir):
        os.mkdir(motTxt_out_dir)
    
    for file in os.listdir(dir_in):
        if '_smooth.txt' in file:
            q.put(file)

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
