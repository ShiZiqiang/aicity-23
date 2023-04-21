import sys
sys.path.append('../')
from deep_sort_app import *
from dyh.dyh_generate_detections import list_valVideo, list_testVideo

if __name__ == '__main__':
    '''VIRAT'''
    # frames_root = '/home/spring/virat_ActEV/virat_frame'
    # detections_dir = '/data1/dyh/Results/ActEV2021/TEST/DET/DetectionFinal_features'
    # output_dir = '/data1/dyh/Results/ActEV2021/TEST/MOT/Final_deepsort'
    # list_video = list_testVideo
    #
    # for i, video in enumerate(list_video, start=1):
    #     # if i != 3: continue
    #     # if video != 'VIRAT_S_010000_02_000388_000421.mp4': continue
    #     print('{}/{} processing {}...'.format(i, len(list_video), video))
    #     for class_id in [1, 2]:
    #         if not os.path.exists(output_dir):
    #             os.mkdir(output_dir)
    #         frames_dir = os.path.join(frames_root, video)
    #         detections_path = os.path.join(detections_dir, video[:-4] + '.npy')
    #         output_path = os.path.join(output_dir, video + '_class%d'%class_id + '.txt')
    #         run(
    #             sequence_dir=frames_dir,
    #             detection_file=detections_path,
    #             output_file=output_path,
    #             min_confidence=0.5,  # 置信度阈值
    #             nms_max_overlap=1.0,
    #             min_detection_height=0,
    #             max_cosine_distance=0.4,  # 0.2,
    #             nn_budget=100,
    #             display=False,
    #             class_id=class_id,
    #         )
    '''MEVA'''
    import glob
    from os.path import join, split, exists
    frame_root = '/disk2/meva_data/meva_frames'
    detections_dir = '/home/sugar/workspace/train_data'
    output_dir = detections_dir
    for i, file in enumerate(glob.glob(join(detections_dir, '*.npy')), start=1):
        if exists(file.replace('.npy', '_mot.txt')): continue
        print('processing the {}th file {}'.format(i, file))
        video = split(file)[1][:-4] + '.avi'
        run(
            sequence_dir=join(frame_root, video),
            detection_file=file,
            output_file=file.replace('.npy', '_mot.txt'),
            min_confidence=0.3,
            nms_max_overlap=1.0,
            min_detection_height=0,
            max_cosine_distance=0.4,
            nn_budget=100,
            display=False,
        )
