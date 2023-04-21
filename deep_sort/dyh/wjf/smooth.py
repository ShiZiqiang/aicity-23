"""
@Author: Du Yunhao
@FileName: smoothTracks.py
@Contact: dyh_bupt@163.com
@Time: 2020/9/11 21:20
@Discription: 基于参数GPR算法的轨迹平滑

"""
import os
import time
import numpy as np
from os.path import exists
from datetime import datetime
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, ConstantKernel as C

# from preprocess import *

gp_kernel = RBF(length_scale=100, length_scale_bounds='fixed')
def run_2(q):
    while True:
        get = q.get()

        if isinstance(get, type(-1)):
            break
        try:
            
            smooth_the_motTxt(
                motTxt_in_dir=r'//home/sugar/workspace/train_data',
                motTxt_out_dir=motTxt_out_dir,
                file=get
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

"""轨迹平滑实验：GPR vs 参数方程GPR"""
def smooth_one_track(track_in, rbf_scale=100, rbf_bounds='fixed'):
    track_out = np.zeros(shape=track_in.shape, dtype=track_in.dtype)
    x, y = track_in[:, 0].reshape(-1, 1), track_in[:, 1].reshape(-1, 1)
    index = np.array([i for i in range(x.shape[0])]).reshape(-1, 1)
    x_gpr, y_gpr = GPR_RBF(index, x, rbf_scale=rbf_scale, rbf_bounds=rbf_bounds), \
                   GPR_RBF(index, y, rbf_scale=rbf_scale, rbf_bounds=rbf_bounds)
    track_out[:, 0], track_out[:, 1] = x_gpr[:, 0], y_gpr[:, 0]
    return track_out

def GPR_RBF(x, y, rbf_scale=1, rbf_bounds=(1e-5, 1e5)):
    """
    :param x: X
    :param y: Y
    :param rbf_scale: RBF length_scale
    :param rbf_bounds: RBF length_scale_bounds，(a, b) or 'fixed'
    :return: y_gpr
    """
    gp_kernel = RBF(length_scale=rbf_scale, length_scale_bounds=rbf_bounds)
    gpr = GaussianProcessRegressor(kernel=gp_kernel)
    gpr.fit(x, y)
    y_gpr = gpr.predict(x, return_std=False)
    return y_gpr

def run_gpr_and_visualize(track, figure_name='tracks'):
    """
    :param track: ndarray, 轨迹信息
    """
    '''获取数据'''
    x, y = track[:, 0].reshape(-1, 1), track[:, 1].reshape(-1, 1)
    index = np.array([i for i in range(x.shape[0])]).reshape(-1, 1)
    '''设置参数'''
    rbf_scale1, rbf_bounds1 = 1, 'fixed'
    rbf_scale2, rbf_bounds2 = 100, 'fixed'
    '''运行GPR'''
    x_gpr1, y_gpr1 = x, GPR_RBF(x, y, rbf_scale=rbf_scale1, rbf_bounds=rbf_bounds1)
    x_gpr2, y_gpr2 = GPR_RBF(index, x, rbf_scale=rbf_scale2, rbf_bounds=rbf_bounds2), \
                     GPR_RBF(index, y, rbf_scale=rbf_scale2, rbf_bounds=rbf_bounds2)
    '''可视化'''
    plt.figure(figure_name)
    # 1. 传统GPR
    plt.subplot(2, 2, 1)
    plt.title('传统GPR：lambda={}-{}'.format(rbf_scale1, rbf_bounds1))
    plt.axis('equal')
    plt.scatter(x, y, s=5, c='b')
    plt.plot(x_gpr1, y_gpr1, color='red', marker=None, linestyle='--')
    # 2. 参数方程GPR
    plt.subplot(2, 2, 2)
    plt.title('参数方程GPR：lambda={}-{}'.format(rbf_scale2, rbf_bounds2))
    plt.axis('equal')
    plt.scatter(x, y, s=5, c='b')
    plt.plot(x_gpr2, y_gpr2, color='red', marker=None, linestyle='--')
    # 3. 参数方程GPR—x(t)
    plt.subplot(2, 2, 3)
    plt.title('参数方程GPR：x(t)')
    plt.scatter(index, x, s=5, c='b')
    plt.plot(index, x_gpr2, color='red', marker=None, linestyle='--')
    # 4. 参数方程GPR—y(t)
    plt.subplot(2, 2, 4)
    plt.title('参数方程GPR：y(t)')
    plt.scatter(index, y, s=5, c='b')
    plt.plot(index, y_gpr2, color='red', marker=None, linestyle='--')

    plt.tight_layout()
    plt.show()

"""对motTxt做平滑"""
def smooth_the_motTxt(motTxt_in_dir, motTxt_out_dir, file):
    
    print('processing file {}...'.format(file))
    motTxt_in_path = os.path.join(motTxt_in_dir, file)
    motTxt_out_path = os.path.join(motTxt_out_dir, file).replace('_mot.txt', '_smooth.txt')
    array_tracks_in = np.loadtxt(motTxt_in_path, delimiter=',')
    array_tracks_out = np.empty([0, 10])
    set_objectID = set(array_tracks_in[:, 1])
    # 遍历每个目标的轨迹
    for j, objectID in enumerate(set_objectID, start=1):
        # print('  processing object {}/{}...'.format(j, len(set_objectID)))
        track = array_tracks_in[array_tracks_in[:,1]==objectID]
        # if track.shape[0] <= 10:  # 删除过短轨迹
        #     continue
        index, x, y, w, h = track[:, 0].reshape(-1, 1), track[:, 2].reshape(-1, 1), \
                            track[:, 3].reshape(-1, 1), track[:, 4].reshape(-1, 1), track[:, 5].reshape(-1, 1),
        gpr_x, gpr_y, gpr_w, gpr_h = GaussianProcessRegressor(kernel=gp_kernel), \
                                        GaussianProcessRegressor(kernel=gp_kernel), \
                                        GaussianProcessRegressor(kernel=gp_kernel), \
                                        GaussianProcessRegressor(kernel=gp_kernel)
        gpr_x.fit(index, x)
        gpr_y.fit(index, y)
        gpr_w.fit(index, w)
        gpr_h.fit(index, h)
        x_gpr = np.clip(gpr_x.predict(index, return_std=False), a_min=0, a_max=1920)
        y_gpr = np.clip(gpr_y.predict(index, return_std=False), a_min=0, a_max=1080)
        w_gpr = np.clip(gpr_w.predict(index, return_std=False), a_min=0, a_max=1920)
        h_gpr = np.clip(gpr_h.predict(index, return_std=False), a_min=0, a_max=1080)
        array_tracks_out = np.append(
            array_tracks_out,
            np.concatenate((
                index,
                np.array([objectID]).repeat(index.shape[0]).reshape(-1,1),
                x_gpr, y_gpr, w_gpr, h_gpr,
                track[:, 6:10]
            ), axis=1),
            axis=0
        )
    np.savetxt(motTxt_out_path, array_tracks_out, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
    # break


if __name__ == '__main__':
    """轨迹平滑实验：GPR vs 参数方程GPR"""
    # track = np.loadtxt(
    #     # r'D:\workspace\Competition\TRECVID 2020 ActEV\dyhActevScripts\files\leftTurn_263_vehicle_000000_2594.txt',
    #     r'D:\Data\ActEV\VehicleOnlyTracks\train_txt\LeftTurn\leftTurn_4109_construction_vehicle_050000_04_000640_000690_1455.txt',
    #     delimiter=','
    # )
    # track_1 = track[:, :2]
    # track_4 = preprocess_one_track(track_1, visualize=False, figure_save_path=False)
    # # track_4 = track_1
    #
    # run_gpr_and_visualize(track_4)
    # txt_in_dir = r'D:\Data\ActEV\VehicleOnlyTracks\train_txt\LeftTurn'
    # for f in os.listdir(txt_in_dir):
    #     file = os.path.join(txt_in_dir, f)
    #     track = np.loadtxt(file, delimiter=',')
    #     track_1 = track[:, :2]
    #     track_4 = preprocess_one_track(track_1, False, False)
    #     run_gpr_and_visualize(track_4, figure_name=f.replace('.txt', ''))

    q = Queue()
    process_num = 60
    process_pool = []
    motTxt_out_dir = r'/home/sugar/workspace/train_data'
    if not exists(motTxt_out_dir):
        os.mkdir(motTxt_out_dir)
    
    for file in os.listdir('/home/sugar/workspace/train_data'):
        if '_mot.txt' in file:
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
