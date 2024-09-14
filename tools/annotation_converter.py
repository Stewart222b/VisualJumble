from abc import ABC, abstractmethod
from utils import *

class Converter(ABC):
    def __init__(self, src_path, target_path, **kwargs):
        """
        args:
            src_path: The source file path
            target_path: The target file path

        """
        self.src_path = src_path
        self.target_path = target_path

    def check_path(self):
        '''
        检查路径是否有效
        
        '''
        print(f"Checking paths: {self.src_path} -> {self.target_path}")
        if not self.src_path or not self.target_path:
            raise ValueError("Source or target path is not provided")

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
        读取源文件
        read source file
        """
        pass

    @abstractmethod
    def _transform(self, data):
        """
        转换数据
        transform data
        """
        pass

    @abstractmethod
    def _write(self, data):
        """
        写入目标文件
        
        """
        pass


class TxtConverter(Converter):
    def _read(self):
        print("Reading from TXT")
        # 读取 txt 文件的逻辑
        return "txt data"

    def _transform(self, data):
        print("Transforming TXT data")
        # 将 txt 数据转换为其他格式的逻辑
        return f"transformed {data}"

    def _write(self, data):
        print(f"Writing data to target: {self.target_path}")
        # 写入目标文件的逻辑


class JsonConverter(Converter):
    def _read(self):
        print("Reading from JSON")
        # 读取 json 文件的逻辑
        return "json data"

    def _transform(self, data):
        print("Transforming JSON data")
        # 将 json 数据转换为其他格式的逻辑
        return f"transformed {data}"

    def _write(self, data):
        print(f"Writing data to target: {self.target_path}")
        # 写入目标文件的逻辑


class XmlConverter(Converter):
    def _read(self):
        print("Reading from XML")
        # 读取 xml 文件的逻辑
        return "xml data"

    def _transform(self, data):
        print("Transforming XML data")
        # 将 xml 数据转换为其他格式的逻辑
        return f"transformed {data}"

    def _write(self, data):
        print(f"Writing data to target: {self.target_path}")
        # 写入目标文件的逻辑


def main():
    cvtr = TxtConverter('/rt/src', '/rt/target')

    cvtr.test()

    print(cvtr.src_path)