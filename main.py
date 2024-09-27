import argparse
import sys
from app.clarity import module
from pathlib import Path

FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple tool of calculating sharpness")
    # parameters
    parser.add_argument('--hist', action='store_true', help='show the statistics as a histogram')
    parser.add_argument('--source', type=str, default=ROOT / 'data/', help='<image_path> or <video_path>')
    parser.add_argument('--save-csv', action='store_true', help='save results to result.csv')
    parser.add_argument('--save-dir', type=str, default= ROOT / 'result', help='save the results to <save_dir>')
    parser.add_argument('--size', type=int, default=(1920, 1080), help='save results to result.csv')
    parser.add_argument('-v', '--visualize', action='store_true', default=True, help='visualize clarity on the source, especially useful for video inputs')
    parser.add_argument('-f', '--fps', type=int, default=2, help='fps of visualization, only valid when used with <-v> and a video source')

    # arguments
    args = parser.parse_args()
    clarity_tool = module.sharpness.Sharpness(args)

    clarity_tool.get() # get the clarity