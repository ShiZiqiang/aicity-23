# -*- coding: utf-8 -*-
import subprocess
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames')
    parser.add_argument('--out_folder', help='the dir to save frames')
    args = parser.parse_args()
    if args.out_folder is None:
        raise ValueError('--out_folder is None')
    return args


if __name__ == '__main__':
    args = parse_args()

    path = './test_videos'
    save_path = args.out_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(path):
        if file.endswith('.txt'):
            with open(os.path.join(path, file), 'r') as f:
                for l in f:
                    l = l.strip()
                    video_id, video_name = l.split(" ")
                    if not os.path.exists(os.path.join(save_path, video_id)):
                        os.makedirs(os.path.join(save_path, video_id))                
                    command = 'ffmpeg -i {} -q:v 2 -f image2 {}/img_%06d.jpg'.format(os.path.join(path, video_name), os.path.join(save_path, video_id))
                    print(command)
                    p = subprocess.Popen(command, shell=True)
                    p.wait()
