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
from strong_sort.strong_sort_app import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test net')
    parser.add_argument('--input_folder', help='the frames path')
    parser.add_argument('--out_file', help='the dir to save results')
    parser.add_argument('--detector', help='detector path', default="checkpoints/detectors_cascade_rcnn.pth")
    parser.add_argument('--feature', help='feature path', default="checkpoints/feature.pth")
    parser.add_argument('--b2', help='b2 path', default="checkpoints/b2.pth")
    parser.add_argument('--resnest50', help='r50 path', default="checkpoints/s50.pth")
    parser.add_argument('--resnest101', help='r101 path', default="checkpoints/s101.pth")

    args = parser.parse_args()
    if args.input_folder is None:
        raise ValueError('--input_folder is None')
    if args.out_file is None:
        raise ValueError('--out_file is None')  
    return args

def compute_area(rec1, rec2, thr):
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        if S1 < S2:
            if S_cross/S1 > thr:
                return rec1
            else:
                return 0
        else:
            if S_cross/S2 > thr:
                return rec2 
            else:
                return 0

def find_traywohand(model, frame):

    result = inference_detector(model, frame)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    if len(bbox_result[0]) == 0:
        return None
    # white tray
    if len(bbox_result[61]) == 0:
        return None
    tray_bboxes = np.vstack(bbox_result[61])
    index_trays = np.argsort(tray_bboxes[:, -1])
    # person
    person_bboxes = np.vstack(bbox_result[0])
    person_indices = person_bboxes[:, -1] > 0.1
    person_segms = segm_result[0]
    person_segms = np.stack(person_segms, axis=0)
    person_segms = person_segms[person_indices]      # 1*1080*1920
    person_bboxes = person_bboxes[person_indices]
    # find tray without hand
    if index_trays.size:
        index_tray = index_trays[-1]
        bbox_tray = tray_bboxes[index_tray]
        if bbox_tray[-1] > 0.5:
            w = bbox_tray[2] - bbox_tray[0]
            h = bbox_tray[3] - bbox_tray[1]
            if (400 < bbox_tray[0] < 900) and (200 < bbox_tray[1] < 600) and (1000 < bbox_tray[2] < 1500) and (600 < bbox_tray[3] < 1000) and (700*500 <w*h < 900 * 700):
                for person_segm in person_segms:
                    for i in range(int(bbox_tray[0]), int(bbox_tray[2])):
                        for j in range(int(bbox_tray[1]), int(bbox_tray[3])):
                            if person_segm[j][i]:
                                return None
                return bbox_tray
    return None

def person_seg(model, frame):
    results = inference_detector(model, frame)
    if isinstance(results, tuple):
        bbox_result, segm_result = results
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = results, None
    if len(bbox_result[0]) == 0:
        return None
    segms = segm_result[0]
    bboxes = np.vstack(bbox_result[0])
    indices = bboxes[:, -1] > 0.1
    segms = np.stack(segms, axis=0)
    segms = segms[indices]
    if len(segms) == 0:
        return None
    return segms

def crop(frame, tray):
    frame = frame[int(tray[1]):int(tray[3])][:][:]
    frame = np.transpose(frame, (1, 0, 2))
    frame = frame[int(tray[0]):int(tray[2])][:][:]       
    frame = np.transpose(frame, (1, 0, 2))

    return frame

def maskhand(frame, tray, segms, maskframe):
    # crop segms
    masks_per_frame = []
    for mask in segms:
        mask = mask[int(tray[1]):int(tray[3])][:]
        mask = np.transpose(mask, (1, 0))
        mask = mask[int(tray[0]):int(tray[2])][:]
        mask_per_frame = np.array([[0] * (int(tray[2]) - int(tray[0]))] * (int(tray[3]) - int(tray[1])))        
        mask = np.transpose(mask, (1, 0))
        masks_per_frame.append(mask)
    for mask in masks_per_frame:
        mask = np.array(mask, dtype=np.int8)
        mask_per_frame = mask_per_frame | mask
    # mask
    mask_part = []
    other_part = []
    first_frame_trans = np.transpose(maskframe, (2, 0, 1))
    frame = np.transpose(frame, (2, 0, 1))
    for c in range(3):
        mask_part.append(np.multiply(first_frame_trans[c][:][:], mask_per_frame))
        other_part.append(np.multiply(frame[c][:][:], ~ (mask_per_frame.astype(np.bool_))))
    frame = np.add(mask_part, other_part)
    frame = np.transpose(frame, (1, 2, 0))
    img = np.array(frame, dtype=np.uint8)
    # img = Image.fromarray(img)

    return img

def getscore(model, video_id, img, list_det):
    roi = img[int(list_det[3]):int(list_det[3] + list_det[5]), int(list_det[2]):int(list_det[2] + list_det[4])]
    # cv2.imwrite(os.path.join('./crop_test/2', 'img_%06d.jpg'%(list_det[0])), roi)
    result, scores, features = inference_model(model, roi)
    line_new = [video_id] + list(list_det[:6]) + list(scores[0])
    return line_new

def getfeature(model, img, list_det):
    roi = img[int(list_det[3]):int(list_det[3] + list_det[5]), int(list_det[2]):int(list_det[2] + list_det[4])]
    result, scores, features = inference_model(model, roi)
    # line_new = list(list_det[:9]) + [-1] + features[0].detach().cpu().numpy().tolist()[0]
    line_new = features[0].detach().cpu().numpy()[0]
    return line_new

# 跟踪
def track(scores):
    # input
    # scores: [video_id, fid, score, x, y, w, h, cls_score*117, feature*1280] *n
    # output
    # scores: [video_id, fid, track_id, x, y, w, h, score*117] *n


    scores = run(
        detections=scores,
        min_confidence=0.3,
        nms_max_overlap=1.0,
        min_detection_height=0,
        max_cosine_distance=0.4, # 0.5
        nn_budget=100,
        display=False,
    )

    return scores

# init
args = parse_args()
# pretrain detector
pretrain_config_file = 'mmdetection/configs/detectors/detectors_htc_r50_1x_coco.py'
pretrain_checkpoint_file = 'checkpoints/detectors_htc_r50_1x_coco-329b1453.pth'
pretrain_detector = init_detector(pretrain_config_file, pretrain_checkpoint_file)
# detector
config_file = 'mmdetection/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py'
checkpoint_file = args.detector
detector = init_detector(config_file, checkpoint_file)
# feature model
config_file = 'configs/efficientnet-b0_8xb32-01norm_in1k.py'
checkpoint_file = args.feature
feature_model = init_model(config_file, checkpoint_file)
# model b2
config_file = 'mmclassification/configs/efficientnet/efficientnet-b2_8xb32-01norm_in1k.py'
checkpoint_file = args.b2
model_b2 = init_model(config_file, checkpoint_file)
# model resnest50
config_file = 'mmclassification/configs/resnest/resnest50_32xb64_in1k.py'
checkpoint_file = args.resnest50
model_s50 = init_model(config_file, checkpoint_file)
# model resnest 101
config_file = 'mmclassification/configs/resnest/resnest101_32xb64_in1k.py'
checkpoint_file = args.resnest101
model_s101 = init_model(config_file, checkpoint_file)

def process(video_id):
    # frame path
    # video_id = 3
    print('video_id {}'.format(video_id))
    frame_path = './frames/%d'%(video_id)

    # 找一帧白色托盘无人手
    start_time = time.time()
    white_tray = None
    for fid in range(1, len(os.listdir(frame_path)) + 1, 100):
        frame = os.path.join(frame_path, 'img_%06d.jpg'%(fid))
        frame = cv2.imread(frame)
        result = find_traywohand(pretrain_detector, frame)
        if result is not None:
            mask_frame = crop(frame, result)
            white_tray = result
            break
    if white_tray is None:
        for fid in range(1, len(os.listdir(frame_path)) + 1):
            frame = os.path.join(frame_path, 'img_%06d.jpg'%(fid))
            frame = cv2.imread(frame)
            result = find_traywohand(pretrain_detector, frame)
            if result is not None:
                mask_frame = crop(frame, result)
                white_tray = result
                break
    if white_tray is None:
        white_tray = [500, 250, 1370, 920]
        mask_frame = crop(cv2.imread(os.path.join(frame_path, 'img_0001.jpg')), white_tray)
    # 找不到间隔50帧再重新遍历？
    end_time = time.time()
    print('Find time:', end_time - start_time)


    #cv2.imwrite(os.path.join('./debug/', 'mask_frame.jpg'), mask_frame)

    # 分割人手 + 检测
    features = []
    scores = []
    for fid in range(1, len(os.listdir(frame_path)) + 1):
    # for fid in range(750, 770):
        start_time = time.time()
        # crop and mask
        frame = os.path.join(frame_path, 'img_%06d.jpg'%(fid))
        # frame = np.array(Image.open(frame))
        frame = cv2.imread(frame)
        frame_crop = crop(frame, white_tray)

        #cv2.imwrite(os.path.join('./debug/', 'frame_crop_tray_%06d.jpg'%(fid)), frame_crop)
        
        result = person_seg(pretrain_detector, frame)
        if result is not None:
            frame_crop = np.array(maskhand(frame_crop, white_tray, result, mask_frame))
            #cv2.imwrite(os.path.join('./debug/', 'frame_crop_maskhand_%06d.jpg'%(fid)), frame_crop)
        
        # detect
        result = inference_detector(detector, frame_crop)
        #print(result)
        instances = result[0].tolist()
        results = []
       
        # 卡阈值
        for instance in instances:
            if instance[4] >= 0.8: #good original thre
                results.append(instance)
        
       
        # 大框去小框
        rm_list = []
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                rm_instance = compute_area(results[i], results[j], 0.8)
                if rm_instance != 0:
                    rm_list.append(rm_instance)
        for rm_instance in rm_list:
            if rm_instance in results:
                results.remove(rm_instance)

        # -------------------------------
        # classify
        for i in results:
            feature = getfeature(feature_model, frame_crop, [fid, -1, i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, i[4], -1, -1])
            score1 = getscore(model_b2, video_id, frame_crop, [fid, i[4], i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, -1, -1, -1])
            score2 = getscore(model_s50, video_id, frame_crop, [fid, i[4], i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, -1, -1, -1])
            score3 = getscore(model_s101, video_id, frame_crop, [fid, i[4], i[0], i[1], i[2] - i[0] + 1, i[3] - i[1] + 1, -1, -1, -1])
            features.append(feature)
            scores.append(np.hstack((np.mean([score1, score2, score3], axis=0), feature)))
            # scores.append(np.hstack((score1, feature)))
        end_time = time.time()
        print('Finished[%d/%d] cost:%.4fs/frame eta:%dm%ds'%(fid, len(os.listdir(frame_path)), end_time - start_time, \
            divmod((end_time - start_time) * (len(os.listdir(frame_path)) - fid), 60)[0], \
            divmod((end_time - start_time) * (len(os.listdir(frame_path)) - fid), 60)[1]), end='\r')
    print('')

    # 输出feature和scores
    # np.save('features.npy', np.array(features), allow_pickle=False)
    np.savetxt('city23_scores_%d.txt'%(video_id), np.array(scores), fmt='%.2f', delimiter=',')

    # 根据features修正scores
    # results = track(features, scores)
    results = track(scores)
    
    #np.savetxt('1_sorted_cores.txt', np.array(results), fmt='%.2f', delimiter=',')
    results.sort(key=lambda x: x[1])
    #np.savetxt('1_sorted_sorted_cores.txt', np.array(results), fmt='%.2f', delimiter=',')

    # 后处理
    start_time = time.time()
    instances = []
    preds = []
    for result in results:
        if [result[0], result[2]] not in instances:
            instances.append([result[0], result[2]])
    for instance in instances:
        instance_frame = []
        instance_score = []
        for result in results:
            if instance == [result[0], result[2]]:
                instance_frame.append(result[1])
                instance_score.append(list(result[7:]))
        # 连续轨迹前后帧差距过大进行拆分
        idx_list = [0]
        for i in range(len(instance_frame) - 1):
            if instance_frame[i] < instance_frame[i+1] - 30:
                idx_list.append(i+1)
        idx_list.append(len(instance_frame))
        for i in range(len(idx_list) - 1):
            frame_c_list = []
            frame_s_list = instance_score[idx_list[i]:idx_list[i+1]]
            for frame in frame_s_list:
                frame_c_list.append(frame.index(max(frame)) + 1)
            track_c_counter = Counter(frame_c_list)
            track_c = track_c_counter.most_common(1)
            top1_ss = []
            for frame in frame_s_list:
                top1_ss.append(frame[track_c[0][0] - 1])
            mean_score = np.mean(top1_ss)
            if len(instance_frame[idx_list[i]:idx_list[i+1]]) > 5 and track_c[0][0]< 116 and mean_score > 0.25:
                pred = []
                pred.append(int(instance[0]))
                pred.append(track_c[0][0])
                pred.append(int(np.mean(instance_frame[idx_list[i]:idx_list[i+1]])))
                if track_c[0][0] in [35, 37, 99, 106]:
                    print(track_c[0][0])
                    print(mean_score)
                preds.append(pred)
    

    # 同类别结果取平均
    # 同类别结果取最小
    preds_idx = []
    preds_new =[]
    for pred in preds:
        if pred[:2] not in preds_idx:
            preds_idx.append(pred[:2])
            preds_new.append(pred)
        else:
            preds_new[preds_idx.index(pred[:2])].append(pred[2])
    preds_final = []
    for i in range(len(preds_new)):
        if len(preds_new[i][2:]) == 1:
            preds_final.append(preds_new[i])
            continue
        idx_list = [0]
        for j in range(len(preds_new[i][2:])):
            if j != 0:
                if preds_new[i][2+j] - preds_new[i][1+j] > 360:
                    idx_list.append(j)
        idx_list.append(len(preds_new[i][2:]))
        for j in range(len(idx_list) - 1):
            pred = []
            for id in preds_new[i][:2]:
                pred.append(id)
            pred.append(round(np.mean(preds_new[i][2+idx_list[j]:2+idx_list[j+1]])))
            preds_final.append(pred)
    preds = preds_final
    
    # 重新排序
    preds_new = []
    for i in range(5):
        tmp = []
        for pred in preds:
            if pred[0] == i + 1 :
                tmp.append(pred)
        for pred in sorted(tmp, key=lambda x:x[2]):
            preds_new.append(pred)
    # 输出结果
    end_time = time.time()
    print('Post-process cost %fs'%(end_time - start_time))

    print('Finished video%d'%(video_id))

    return preds_new


if __name__ == '__main__':

    frame_path = args.input_folder
    videos = os.listdir(frame_path)
    videos.sort(key=lambda x: int(x))
    for i in range(len(videos)):
        if i > -1:
            preds = process(int(videos[i]))
            with open(args.out_file, 'a+') as f:
            # with open('./test.txt', 'a+') as f:
                for pred in preds:
                    f.write(" ".join(str(i) for i in pred) + '\n')
