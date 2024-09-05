#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Stewart222b
# @Date: 2024-09-02
# @function: Calculate the sharpness of given source

#import argparse
import cv2
import glob
import os
import sys
import matplotlib.pyplot as plt
#from utils.visualize import *
from pathlib import Path

FILE = Path(__file__).resolve()

ROOT = FILE.parents[2]  # root directory
print(str(FILE))
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.visualize import *

class Sharpness:
    def __init__(self, args):
        self.parse_opt(args)
        self.img_format = ['jpg', 'jpeg', 'png']
        self.vid_format = ['mp4', '264', 'h264']
        self.save_dir =  str(self.get_save_dir(args.save_dir))
        self.frames = self.get_frames()
        self.frame_count = self.get_frame_count() 
        self.scores = []

    def parse_opt(self, args):
        print(args)
        self.source = str(args.source[0] if isinstance(args.source, list) else args.source) # source path
        self.visualize = args.visualize # visualize the result
        self.hist = args.hist # show the statistics as a histogram
        self.save_csv = args.save_csv # save the sharpness values to result.csv file
        self.size = args.size # input size
        self.fps = args.fps # fps
        self.threshold = 1000

    # calculate clarity by getting the variance of the laplacian
    def get(self,):
        count = 0
        images = self.frames['imgs']
        videos = self.frames['vids']

        # handle videos
        for vid in videos:
            #cap = cv2.VideoCapture(vid)
            #cap, writer = self.config_writer(vid, self.save_dir + '/' + vid.split('/')[-1])
            cap, writer = config_writer(vid, self.save_dir + '/' + vid.split('/')[-1])

            fps_count = 0
            old_sharpness = None
            while cap.isOpened():
                ret, frame = cap.read()  # read one frame
                if not ret:
                    break  # exit if no frame get
                fps_count += 1
                count += 1
                sharpness = self.calculate_sharpness(frame, vid, count)
                if self.visualize:
                    if (fps_count % self.fps == 0) or (not old_sharpness):
                        old_sharpness = sharpness
                    self.draw_texts(frame, old_sharpness, out=writer, count=count)
            cap.release()

        # handle images
        for img in images:
            frame = cv2.imread(img)
            count += 1
            sharpness = self.calculate_sharpness(frame, img, count)
            if self.visualize:
                self.draw_texts(frame, sharpness, self.save_dir + '/' + img.split('/')[-1])


        if self.hist:
            self.show_hist()
        
        if self.save_csv:
            self.save()

        return 

    # calculate the sharpness of given frame
    def calculate_sharpness(self, frame, file_name, count):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.size)
        sharpness = self.variance_of_laplacian(gray)
        self.scores.append((file_name, sharpness))
        print(f"[{count}/{self.frame_count}] {file_name}: {sharpness}")
        return sharpness

    # calculate the variance of the laplacian
    def variance_of_laplacian(self, src):
        var = cv2.Laplacian(src, cv2.CV_64F).var()
        return round(var, 2)

    def get_frames(self):
        frames = {'imgs':[], 'vids':[]}
        src_files = glob.glob(os.path.join(self.source, '*'))
        #src_files = os.listdir(self.source)
        for file in src_files:
            format = file.split('.')[-1]
            if format in self.img_format:
                frames['imgs'].append(file)
            elif format in self.vid_format:
                frames['vids'].append(file)
            else:
                print(f'Skip invalid file name: {file}')
            
        return frames

    # get the total frame count of given source
    def get_frame_count(self):
        count = 0
        # number of image frame
        count += len(self.frames['imgs'])
        # number of video frame
        videos = self.frames['vids']
        for vid in videos:
            cap = cv2.VideoCapture(vid)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            count += frame_count
        return count
    
    def get_save_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    # show the histogram using Matplotlib
    def show_hist(self):
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist([item[1] for item in self.scores], bins=10, color='blue', edgecolor='black')
        # Add the sum of every column
        for count, x in zip(n, bins):
            plt.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom')

        plt.title('Image Sharpness Histogram')
        plt.xlabel('Sharpness')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def config_writer(self, vid, file_name):
        if not self.visualize:
            return None
        
        cap = cv2.VideoCapture(vid)

        # get width and height
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置视频输出参数
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use MP4 encoding
        out = cv2.VideoWriter(file_name, fourcc, 30.0, (w, h))
        return out
    
    def draw_texts(self, frame, sharpness, img_name=None, out=None, count=None):
        cv2.rectangle(frame, (0, 0), (600, 65), (0, 0, 0), thickness=-1)
        cv2.putText(frame, f'Current clarity: {sharpness}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        cv2.rectangle(frame, (0, 65), (250, 130), (0, 0, 0), thickness=-1)
        if sharpness < self.threshold:
            cv2.putText(frame, f'Not Clear', (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f'Clear', (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
        if out:
            out.write(frame)
        else:
            cv2.imwrite(img_name, frame)
              
    def save(self, file_name='result.csv'):
        with open(file_name, 'a') as file:  # 'a' for append
            for frame, sharpness in self.scores:
                file.write(f"{frame},{sharpness}\n")
                # Run the command and capture the output 
                # import subprocess
                #if  l_threshold < sharpness < r_threshold:
                #    result = subprocess.run([f'imgcat -s -W {size}px -H {size}px {image_path}'], capture_output=False, text=True, shell=True)
                #    print(f"{image_path}: {sharpness}")
        return 