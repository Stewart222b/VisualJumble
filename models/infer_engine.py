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
    parser.add_argument('--img_dir', default='/Users/Larry/Desktop/test_data', help='Test images path 测试图片路径')
    parser.add_argument('--save_dir', default='result', help='Save path 保存路径')
    parser.add_argument('--device', type=str, default='cpu', help=f'Infer device 推理设备')
    parser.add_argument('--model', type=str, default='lprnet', help=f'models/networks 模型/网络: {MODELS}')
    #parser.add_argument('--dataset', type=str, default='fashion_mnist', help=f'Available dataset: {DATASETS}')
    parser.add_argument('--pretrained', type=str, default='/Users/Larry/Projects/LPRNet_Pytorch-master/weights/Final_LPRNet_model.pth', help='pretrained model 预训练模型')
    parser.add_argument('--verbose', action='store_true', help='Show details 显示详细信息')

    # Parse arguments 解析参数
    args = parser.parse_args()

    return args


class InferEngine:
    def __init__(self, **kwargs):
        '''
        '''
        args = parse_args()

        self.img_dir = Path(args.img_dir)
        self.save_dir = Path(args.save_dir)
        self.device = get_device(args.device, kwargs.get('device', None))
        self.model, self.model_info = get_model(args.model, True, kwargs.get('model', None))
        self.pretrained = args.pretrained


    def infer(self,):
        input_size = self.model_info.input_size
        preprocess = self.model_info.pre
        postprocess = self.model_info.post

        # load pretrained model 加载预训练模型
        model = load_model(self.model, self.pretrained, self.device)

        images = self.img_dir.glob('*.jpg') if self.img_dir.is_dir() else [self.img_dir]
        for img_path in images:
            # Load data 加载数据
            image = cv2.imread(img_path)
            
            resized = cv2.resize(image, input_size)

            # Preprocess 预处理
            input = preprocess(resized)

            # Infer 推理
            with torch.no_grad(): # Disable gradient calculation 不计算梯度
                model.eval()
                output = model(input)
            
            # Postprocess 后处理
            results = postprocess(output)
            
            # Save result 保存结果
            for result in results:
                print(Info(f'Infer result 推理结果: {result}'))
                image = cv2ImgAddText(image, result, (0, 0))
                
                cv2.imwrite(self.save_dir / Path(img_path.name), image)