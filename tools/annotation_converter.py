from abc import ABC, abstractmethod
from utils import *
from pathlib import Path
import json
import cv2 

class Converter(ABC):
    def __init__(self, src_path, output_path, **kwargs):
        """
        args:
            src_path: The source file path
            target_path: The target file path

        """
        self.src_path = Path(src_path)
        self.output_path = Path(output_path)

    def check_path(self):
        '''
        Check if a path is valid path
        '''
        print(f"Checking paths: {self.src_path} -> {self.output_path}")
        if not self.src_path or not self.output_path:
            print(Error("Source or target path is not provided"))
            exit()

    def convert(self):
        """
        模板方法，负责执行转换流程。子类将实现具体的转换逻辑
        """
        self.check_path()
        converted_data = self._read()  # 读取源文件
        transformed_data = self._transform(converted_data)  # 转换数据
        self._write(transformed_data)  # 将转换后的数据写入目标文件

    @abstractmethod
    def _read(self):
        """
        read source file
        """
        pass

    @abstractmethod
    def _transform(self, data):
        """
        transform data
        """
        pass

    @abstractmethod
    def _write(self, data):
        """
        write data
        """
        pass


class TxtConverter(Converter):
    """
    convert txt label to other format (i.e. json, xml)
    """
    def _read(self):
        print("Reading from TXT")
        return "txt data"

    def _transform(self, data):
        print("Transforming TXT data")
        return f"transformed {data}"

    def _write(self, data):
        print(f"Writing data to target: {self.output_path}")
        # 写入目标文件的逻辑


class JsonConverter(Converter):
    """
    convert json label to other format (i.e. txt, xml)
    """

    
    coco = \
    {
        'class': 
        {
            'categories':
            {
                'id': int,   # class id | 类别id，
                'name': str, # class name | 类别名称
            }
        },
        'image':
        {
            'images':
            {
                'file_name': str, # image name | 图像名称
                'height': float,  # image height | 图像高度 
                'width': float,   # image width | 图像宽度
                'id': int,        # image id | 图像id
            }
        },
        'labels':
        {
            'annotations':
            {
                'image_id': int,     # image id of label ｜ 标签对应的图像id
                'bbox': list[float], # bounding box | 检测框
                'category_id': int,  # class id | 类别id
            }
        }
    }
    
    bdd100k = \
    {
        'class': 
            None,
        'image':
        {
            'name': str, # image name 
            'labels':
            {
                'category': str, # category name
                'box2d': 
                {
                    "x1": float, 
                    "y1": float, 
                    "x2": float,
                    "y2": float,
                }
            }
        },
        'labels':
            None
    }
    


    def __init__(self, src_path, output_path, **kwargs):
        """
        Initializes the json converter with optional parameters. | 使用可选参数初始化json标注转换器。

        Args:
            src_path (str): Path to original json annotation file. | 原始json标注文件的路径。
            output_path (str): Path to save transformed annotations. | 保存转换后的标注的路径。
            image_path (str, optional): Path of corresponding images in json file. | json文件中对应图像的路径。
            image_path (str, optional): Path of corresponding images in json file. | json文件中对应图像的路径。
        """
        super().__init__(src_path, output_path, **kwargs)
        self.json_path = self.src_path
        self.image_path = Path(kwargs.get('image_path', None))
        self.output_format = kwargs.get('output_format', None)
        self.label_format = kwargs.get('json_format', 'bdd100k')

    def _read(self, f):
        classes = []
        images = []
        labels = []

        print("Reading from JSON")
        json_data = json.load(f)

        if self.label_format == 'coco':
            for data in json_data:
                pass
            pass
        elif self.label_format == 'bdd100k':
            if not self.image_path:
                print(Error('Image path needed if json format is bdd100k!'))
                exit()
            
            for data in json_data:
                pass

        return "json data"

    def transform(self):
        return self._transform()

    def _transform(self):
        info = 'Transforming JSON data to '

        if self.output_format == 'txt':
            print(Info(f'{info}{self.output_format.upper()}'))
            # 加载 COCO 格式的 JSON 文件
            with open(self.json_path, 'r') as f:
                json_data = json.load(f)
                #json_data = self._read(f)

            '''
            txt_classes = ['traffic light',
                           'traffic sign',
                           'car',
                           'pedestrian',
                           'bus',
                           'truck',
                           'rider',
                           'bicycle',
                           'motorcycle',
                           'train',
                           'other vehicle',
                           'other person',
                           'trailer']
            '''

            txt_classes = ['car',
                           'bus',
                           'truck',
                           'rider',
                           'bicycle',
                           'motorcycle',
                           'other vehicle']

            print(len(json_data))

            count = 0
            progress = Progress()
            for data in json_data:
                # handle progress bar
                count += 1
                image_name = data['name']
                image = cv2.imread(self.image_path / image_name)
                progress.progress_bar(count, len(json_data), Info(f'Processing {image_name}'))

                w, h = image.shape[1], image.shape[0]

                json_labels = data['labels'] if 'labels' in data.keys() else None
                if not json_labels:
                    continue
                
                txt_labels = []
                for json_label in json_labels:
                    category = json_label['category']
                    if category not in txt_classes:
                        #print(Error(f'Category {category} not in txt_classes, skipping'))
                        continue
                    else:
                        class_id = txt_classes.index(category)
                    
                    bbox = json_label['box2d']
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

                    txt_label = f'{class_id} {(x1+x2) / 2 / w} {(y1+y2) / 2 / h} {(x2-x1) / w} {(y2-y1) / h}'
                    txt_labels.append(txt_label)

                self._write(txt_labels, image_name)

            print("转换完成，所有标注已保存为单独的 txt 文件！")
        elif self.output_format == 'xml':
            print(Info(f'{info}{self.output_format.upper()}'))
            pass
        else:
            print(Error("不支持的格式转换，仅支持 txt, xml"))
            exit()

        return f"transformed {data}"

    def _write(self, labels, name):
        if type(labels) is list:
            labels = '\n'.join(labels)
        elif type(labels) is not str:
            print(Error("不支持的标注，仅支持 list, str"))
            exit()

        name = name.split('.')[0]
        
        with open(self.output_path / f'{name}.{self.output_format}', 'w') as f:
            f.write(labels)


class XmlConverter(Converter):
    """
    convert xml label to other format (i.e. json, txt)
    """
    def _read(self):
        print("Reading from XML")
        return "xml data"

    def _transform(self, data):
        print("Transforming XML data")
        return f"transformed {data}"

    def _write(self, data):
        print(f"Writing data to target: {self.output_path}")


def main():
    cvtr = TxtConverter('/rt/src', '/rt/target')

    cvtr.convert()

    print(cvtr.src_path)