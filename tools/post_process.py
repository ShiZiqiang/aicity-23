from cv2 import line
from mmcls.apis import inference_model, init_model
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import os
import numpy as np
from collections import Counter
import time
import sys
# from .deep_sort.deep_sort_app import *
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from deep_sort.deep_sort_app import *
import argparse




if __name__ == '__main__':


    results = np.loadtxt('1_sorted_sorted_cores.txt', delimiter=',')
    # 后处理
    start_time = time.time()
    instances = []
    preds = []
    for result in results:
        if [result[0], result[2]] not in instances:
            instances.append([result[0], result[2]])
    print(instances)
    for instance in instances:
        print('instance {}'.format(instance))
        instance_frame = []
        instance_score = []
        for result in results:
            if instance == [result[0], result[2]]:
                instance_frame.append(result[1])
                instance_score.append(list(result[7:]))
        # 连续轨迹前后帧差距过大进行拆分
        idx_list = [0]
        print('instance frame {}'.format(instance_frame))
        for i in range(len(instance_frame) - 1):
            if instance_frame[i] < instance_frame[i+1] - 30:
                idx_list.append(i+1)
        idx_list.append(len(instance_frame))
        print('idx_list {}'.format(idx_list))
        for i in range(len(idx_list) - 1):
            frame_c_list = []
            frame_s_list = instance_score[idx_list[i]:idx_list[i+1]]
            for frame in frame_s_list:
                frame_c_list.append(frame.index(max(frame)) + 1)
            track_c_counter = Counter(frame_c_list)
            track_c = track_c_counter.most_common(1)
            print('track_c_counter {}'.format(track_c_counter))
            print('track_c {}'.format(track_c))
            top1_ss = []
            for frame in frame_s_list:
                top1_ss.append(frame[track_c[0][0] - 1])
            mean_score = np.mean(top1_ss)
            print('mean_score {}'.format(mean_score))
            if len(instance_frame[idx_list[i]:idx_list[i+1]]) > 5 and track_c[0][0]< 116 and mean_score > 0.25:
                pred = []
                pred.append(int(instance[0]))
                pred.append(track_c[0][0])
                pred.append(int(np.mean(instance_frame[idx_list[i]:idx_list[i+1]])/ 60))
                if track_c[0][0] in [35, 37, 99, 106]:
                    print(track_c[0][0])
                    print(mean_score)
                preds.append(pred)
                #print('pred {}'.format(pred))
    

    print('preds after split {}'.format(preds))
    np.savetxt('1_preds_after_split.txt', np.array(preds), fmt='%.2f', delimiter=',')
    # 同类别结果取平均
    preds_idx = []
    preds_new =[]
    for pred in preds:
        if pred[:2] not in preds_idx:
            print('pred[:2] {}'.format(pred[:2]))
            preds_idx.append(pred[:2])
            preds_new.append(pred)
        else:
            preds_new[preds_idx.index(pred[:2])].append(pred[2])

    print('preds_idx {}'.format(preds_idx))
    print('preds_new {}'.format(preds_new))
    
    preds_final = []
    for i in range(len(preds_new)):
        if len(preds_new[i][2:]) == 1:
            preds_final.append(preds_new[i])
            continue
        idx_list = [0]
        for j in range(len(preds_new[i][2:])):
            if j != 0:
                if preds_new[i][2+j] - preds_new[i][1+j] > 6:
                    idx_list.append(j)
        idx_list.append(len(preds_new[i][2:]))
        for j in range(len(idx_list) - 1):
            pred = []
            for id in preds_new[i][:2]:
                pred.append(id)
            pred.append(round(np.mean(preds_new[i][2+idx_list[j]:2+idx_list[j+1]])))
            preds_final.append(pred)
    preds = preds_final
    
    print('preds after mean {}'.format(preds))
    np.savetxt('1_preds_after_mean.txt', np.array(preds), fmt='%.2f', delimiter=',')
    # 重新排序
    preds_new = []
    for i in range(5):
        tmp = []
        for pred in preds:
            if pred[0] == i + 1 :
                tmp.append(pred)
        for pred in sorted(tmp, key=lambda x:x[2]):
            preds_new.append(pred)
    print('preds new {}'.format(preds_new))
    # 输出结果
    # with open('./results%d.txt'%(video_id), 'w') as f:
    #     for result in preds_new:
    #         f.write(" ".join(str(i) for i in result) + '\n')
    end_time = time.time()
    print('Post-process cost %fs'%(end_time - start_time))

    #print('Finished video%d'%(video_id))


