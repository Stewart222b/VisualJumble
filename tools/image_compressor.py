from utils import *
from pathlib import Path
import json
import shutil
import os



class ImageCompressor:
    valid_tool = ['opencv', 'pillow']

    def __init__(self, tool) -> None:
        if tool not in self.valid_tool:
            print(Error(f'No tool choosed, please choose a valid tool: {self.valid_tool}'))
            return -1
        self.tool = tool

    def compress(self, src, quality=70):
        if type(src) == str:
            src = Path(src)
        elif type(src) != Path():
            print(Error(f'Invalid source format, please choose a valid source!'))

        if self.tool == 'opencv':
            print(Info('Using OpenCV | 使用 OpenCV'))
            self._opencv_compressor(src, quality)
        elif self.tool == 'pillow':
            print(Info('Using PIL | 使用 PIL'))
            self._pillow_compressor(src, quality)

        print(Info('Finished! | 已完成！'))

    def _opencv_compressor(self, src: Path, quality):
        import cv2
        if src.is_dir():
            images = list(src.rglob('*.jpg'))
            progress = Progress()
            count = 0
            print(Info(f'Compressing... | 正在压缩...'))
            for jpg in images:
                count += 1
                progress.progress_bar(current=count, total=len(images))
                cv2.imwrite(jpg, cv2.imread(jpg), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            img = cv2.imread(src)
            print(Info('Image passed'))
        pass

    def _pillow_compressor(self, src: Path, quality):
        import PIL
        pass



def main():
    print(Info('main function of image compressor'))