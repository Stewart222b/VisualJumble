import inspect
import sys
import time

class Color:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    UNSET = '\033[0m'

    COLORS = {
        'red': RED,
        'green': GREEN,
        'yellow': YELLOW,
        'blue': BLUE,
    }

    def __new__(cls, text, color):
        color_code = cls.COLORS.get(color, cls.UNSET)
        return f'{color_code}{text}{cls.UNSET}'

class Message(Color):
    debug_mode = False

    def __new__(cls, **kwargs):
        '''
        kwargs:
            text
            color
            prefix
            ...
        '''
        message = ''

        text = kwargs.get('text', None)
        color = kwargs.get('color', 'red')
        prefix = kwargs.get('prefix', None)

        # print location of the message if needed
        if cls.debug_mode:
            print(f'[Location] {cls.get_location(inspect.currentframe())}')

        if prefix:
            message = f'[{prefix}] {text}'
        else:
            message = f'{text}'
        
        return Color(message, color)

    @classmethod
    def class_name(cls):
        return cls.__name__
    
    def get_location(frame):
        file = frame.f_back.f_back.f_code.co_filename
        line = frame.f_back.f_back.f_lineno
        return f'File: {file}, line {line}'

class Error(Message):
    _error_color = 'red'

    def __new__(cls, error_message):
        #Message.get_location(inspect.currentframe())
        return super(Error, cls).__new__(cls, text=error_message, color=cls._error_color, prefix=cls.__name__)


class Warning(Message):
    _warning_color = 'yellow'
    
    def __new__(cls, warning_message):
        return super(Warning, cls).__new__(cls, text=warning_message, color=cls._warning_color, prefix=cls.__name__)
    

class Info(Message):
    _info_color = 'green'
    
    def __new__(cls, info_message):
        return super(Info, cls).__new__(cls, text=info_message, color=cls._info_color, prefix=cls.__name__)
    
class Progress(Message):
    def __init__(self):
        self._info_color = 'blue'
        self.bar_base = '█▓▒░' # 100% filled, 75% filled, 50% filled, 25% filled
    
    def __new__(cls):
        return super(Color, cls).__new__(cls)
    
    def progress_bar(self, current: int, total: int, text = None, bar_length=40):
        # 计算完成比例
        fraction = current / total
        
        # 填充进度条
        filled_length = int(bar_length * fraction)
        bar = self.bar_base[0] * filled_length + '-' * (bar_length - filled_length)
        
        # 打印进度条
        percent = fraction * 100
        #sys.stdout.write(f'')
        sys.stdout.write(f'\r{text}\n|{bar}| {percent:.2f}% Complete')
        sys.stdout.flush()  # 刷新输出