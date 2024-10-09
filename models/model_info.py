from abc import ABC, abstractmethod

class ModelInfo(ABC):
    '''
    Model's basic information, including input size, pretrained model, preprocess, postprocess, etc. 
    模型基础信息，包括输入形状，预训练模型，预处理，后处理等。
    '''
    def __init__(self, input_size, **kwargs) -> None:
        self.input_size = input_size
        self.pretrained = kwargs.get('pretrained', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.lr = kwargs.get('lr', None)
        self.criterion = kwargs.get('criterion', None)
        
        self.preprocess = self.pre
        self.postprocess = self.post
    
    
    @abstractmethod
    def pre(self, input):
        '''
        Model's input preprocess 模型输入预处理

        Args:
            input: model's input 模型输入
        Return:
            input: model's input after preprocess 模型预处理过的输入
        '''
        return input
    

    @abstractmethod
    def post(self, output):
        '''
        Model's output postprocess 模型输出后处理

        Args:
            output: model's output 模型输出
        Return:
            output: model's output after postprocess 模型后处理过的输出
        '''
        return output