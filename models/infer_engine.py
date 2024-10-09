import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import *
import torch
from torch.optim import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Inference parameters 推理参数')

    # Add arguments 添加参数
    parser.add_argument('--img_size', default=[94, 24], help='Input size 输入大小')
    parser.add_argument('--img_dir', default="/Users/Larry/Downloads/CBLPRD-330k_v1/CBLPRD-330k/000000035.jpg", help='the test images path')
    parser.add_argument('--device', type=str, default='cpu', help=f'Infer device 推理设备')
    parser.add_argument('--model', type=str, default='lprnet', help=f'models/networks 模型/网络: {MODELS}')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help=f'Available dataset: {DATASETS}')
    parser.add_argument('--pretrained', type=str, default='/Users/Larry/Projects/LPRNet_Pytorch-master/weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    # Parse arguments 解析参数
    args = parser.parse_args()

    return args


class InferEngine:
    def __init__(self, **kwargs):
        '''
        '''
        args = parse_args()

        self.device = get_device(args.device, kwargs.get('device', None))
        self.model, self.model_info = get_model(args.model, True, kwargs.get('model', None))
        self.pretrained = args.pretrained
        self.img_dir = Path(args.img_dir)


    def infer(self,):
        input_size = self.model_info.input_size
        preprocess = self.model_info.pre
        postprocess = self.model_info.post

        # load pretrained model
        if self.pretrained:
            self.model.load_state_dict(torch.load(self.pretrained, map_location=self.device))
            print(Info("Load pretrained model successfully!"))
        else:
            print(Error("Can't found pretrained mode, please check!"))
            return False

        test_data = read_image(self.img_dir, preprocess, size=input_size).to(self.device)

        with torch.no_grad(): # 不计算梯度
            self.model.eval()
            out = self.model(test_data)
        
        out = postprocess(out)
        
        for o in out:
            img = cv2.imread('/Users/Larry/Downloads/CBLPRD-330k_v1/CBLPRD-330k/000000035.jpg')
            img = cv2ImgAddText(img, o, (0, 0))
            
            cv2.imwrite('result/plate.jpg', img)