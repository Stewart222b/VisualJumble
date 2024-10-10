from abc import ABC, abstractmethod

class ModelInfo(ABC):
    """
    Model's basic information, including input size, pretrained model, preprocess, postprocess, etc.
    模型基础信息，包括输入形状，预训练模型，预处理，后处理等。

    Attributes:
        input_size (tuple): Input shape of the model in (height, width) format.
                            模型的输入形状，以（高度，宽度）格式表示。
        pretrained (optional): Path to the pretrained model or pretrained model identifier.
                               预训练模型的路径或标识符。
        optimizer (optional): Optimizer used for model training.
                              用于模型训练的优化器。
        lr (optional): Learning rate for the optimizer.
                       优化器的学习率。
        criterion (optional): Loss function used for model training.
                              用于模型训练的损失函数。
    """

    def __init__(self, input_size: tuple, **kwargs) -> None:
        """
        Initialize the model's basic information.
        初始化模型的基础信息。

        Args:
            input_size (tuple): Input shape of the model in (height, width) format.
                                模型的输入形状，以（高度，宽度）格式表示。
            **kwargs: Optional keyword arguments for additional model configuration, such as
                      'pretrained', 'optimizer', 'lr', and 'criterion'.
                      可选的关键字参数，用于额外的模型配置，如 'pretrained', 'optimizer', 'lr' 和 'criterion'。
        
        Attributes:
            input_size (tuple): Input shape of the model in (height, width) format.
                                模型的输入形状，以（高度，宽度）格式表示。
            pretrained (optional): Path to the pretrained model or pretrained model identifier.
                                   预训练模型的路径或标识符。
            optimizer (optional): Optimizer used for model training.
                                  用于模型训练的优化器。
            lr (optional): Learning rate for the optimizer.
                           优化器的学习率。
            criterion (optional): Loss function used for model training.
                                  用于模型训练的损失函数。
        """
        
        self.input_size = input_size # (h,w)
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