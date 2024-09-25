from utils import *
from pathlib import Path
import json
import shutil
import os
import cv2 

class ImageCompressor:
    valid_tool = ['opencv', 'pillow']

    def __init__(self, tool) -> None:
        if tool not in self.valid_tool:
            print(Error(f'No tool choosed, please choose a valid tool: {self.valid_tool}'))
            return -1
        self.tool = tool

    def compress(self, src):
        if self.tool == 'opencv':
            self._opencv_compressor()
        elif self.tool == 'pillow':
            self._pillow_compressor()

    def _opencv_compressor(self,):
        pass

    def _pillow_compressor(self,):
        pass    


def main():
    print(Info('main function of image compressor'))