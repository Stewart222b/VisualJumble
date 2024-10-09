DEVICES = ['cpu', 'cuda', 'mps']
MODELS = [
    'lenet', # LeNet
    'alexnet', # AlexNet
    'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19', # VGG: 11, 13, 16, 19 ('vgg' will use 'vgg16')
    'lprnet',
]
DATASETS = ['fashion_mnist']
OPTIMIZERS = ['sgd']
CRITERIONS = ['ce', 'bce']

# Chinese license plate chars
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}