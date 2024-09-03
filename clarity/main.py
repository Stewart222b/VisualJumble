import argparse
from sharpness import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple tool of calculating sharpness")
    # parameters
    parser.add_argument('--hist', action='store_true', help='show the statistics as a histogram')
    parser.add_argument('--source', type=str, default=ROOT / 'data/', help='<image_path> or <video_path>')
    parser.add_argument('--save-csv', action='store_true', help='save results to result.csv')
    parser.add_argument('--save-dir', type=str, default= ROOT / 'result', help='save the results to <save_dir>')
    parser.add_argument('--size', type=int, default=(640, 640), help='save results to result.csv')
    parser.add_argument('-v', '--visualize', action='store_true', default=True, help='visualize clarity on the source, especially useful for video inputs')
    parser.add_argument('-f', '--fps', type=int, default=2, help='fps of visualization, only valid when used with <-v> and a video source')

    # arguments
    args = parser.parse_args()
    obj = Sharpness(args)

    obj.get()


    '''
    # source
    src_path = args.source
    print(src_path)
    #src_files = glob.glob(src_path)
    print(src_files)

    scores = []
    score = 0
    for file in src_files:
        file_format = file.split('.')[-1]
        image_format = ['jpg', 'jpeg', 'png']
        video_format = ['mp4', '264']
        if file_format in image_format:
            frame = cv2.imread(file)
            score = calc_clarity(frame)
        elif file_format in video_format:
            cap = cv2.VideoCapture(file)  # 使用摄像头则用 0 或其他摄像头编号
            while cap.isOpened():
                ret, frame = cap.read()  # 读取一帧
                if not ret:
                    break  # 如果没有读取到帧，退出循环

                score = calc_clarity(frame, file)
        else:
            print('Invalid format! Please check input format')
        scores.append([file, score])

    # 按清晰度从高到低排序
    scores.sort(key=lambda x: x[1], reverse=True)
    filename = "sharpness.txt"
    l_threshold = 340
    r_threshold = float('inf')
    with open(filename, 'a') as file:  # 'a' 模式表示追加写入
        for image_path, sharpness in scores:
            file.write(f"{sharpness}\n")
            print(f"{image_path}: {sharpness}")
            # 执行命令并捕获输出
            #if  l_threshold < sharpness < r_threshold:
            #    result = subprocess.run([f'imgcat -s -W {size}px -H {size}px {image_path}'], capture_output=False, text=True, shell=True)
            #    print(f"{image_path}: {sharpness}")
    
    #visualize([s for _, s in sorted_images])'''

