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

    obj.get() # get the clarity