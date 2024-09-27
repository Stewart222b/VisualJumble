import inspect
import time

# Colors | 颜色
RED = '\033[31m' # red color | 红色
GREEN = '\033[32m' # green color | 绿色
YELLOW = '\033[33m' # yellow color | 黄色
BLUE = '\033[34m' # blue color | 蓝色
UNSET = '\033[0m' # reset color | 重置颜色

# ANSI 
UP_ONE_LINE = '\033[F' # move cursor up one line | 光标上移一行
DOWN_ONE_LINE = '\033[B' # move cursor down one line | 光标下移一行
CLEAR_LINE = '\033[K' # clear line | 清除一行

class Color:
    """
    A class to handle color formatting of text using ANSI codes. | 处理文本颜色格式化的类，使用ANSI代码。

    Attributes:
        COLORS (dict): A dictionary mapping color names to ANSI escape codes. | 将颜色名称映射到ANSI转义码的字典。

    Methods:
        __new__(cls, text, color): Return the formatted text with the specified color. | 返回带有指定颜色的格式化文本。
    """

    COLORS = {
        'red': RED,
        'green': GREEN,
        'yellow': YELLOW,
        'blue': BLUE,
    }

    def __new__(cls, text, color):
        """
        Formats the given text with the specified color. | 使用指定颜色格式化给定文本。

        Args:
            text (str): The text to be formatted. | 需要格式化的文本。
            color (str): The color to be applied. | 需要应用的颜色。

        Returns:
            str: The color formatted string. | 返回格式化后的颜色字符串。
        """
        color_code = cls.COLORS.get(color, UNSET)
        return f'{color_code}{text}{UNSET}'

class Message(Color):
    """
    A message class that supports debug mode and colored output. | 支持调试模式和彩色输出的消息类。

    Attributes:
        debug_mode (bool): A flag to determine if debug mode is enabled. | 标记是否启用调试模式。

    Methods:
        __new__(cls, **kwargs): Returns a color formatted message with an optional prefix. | 返回带有可选前缀的彩色格式消息。
        class_name(cls): Returns the name of the class. | 返回类名。
        get_location(frame): Retrieves the file and line number of the message. | 获取消息的文件和行号。
    """

    debug_mode = False

    def __new__(cls, **kwargs):
        """
        Creates a formatted message with optional prefix and color. | 创建带有可选前缀和颜色的格式化消息。

        Args:
            text (str): The message text. | 消息文本。
            color (str): The color of the text (default is 'red'). | 文本颜色（默认为红色）。
            prefix (str, optional): An optional prefix to the message. | 消息的可选前缀。

        Returns:
            str: A color formatted message with optional prefix. | 返回带有可选前缀的彩色格式消息。
        """
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
        """
        Returns the name of the class. | 返回类名。

        Returns:
            str: The class name. | 类名。
        """
        return cls.__name__
    
    def get_location(frame):
        """
        Retrieves the file name and line number from the frame object. | 从frame对象中获取文件名和行号。

        Args:
            frame: The current frame object. | 当前frame对象。

        Returns:
            str: The file name and line number in the format 'File: {file}, line {line}'. | 返回格式为'File: {file}, line {line}'的文件名和行号。
        """
        file = frame.f_back.f_back.f_code.co_filename
        line = frame.f_back.f_back.f_lineno
        return f'File: {file}, line {line}'

class Error(Message):
    """
    A class for handling error messages with a default red color. | 处理带有默认红色的错误消息的类。

    Methods:
        __new__(cls, error_message): Returns a formatted error message with the class name as prefix. | 返回带有类名前缀的格式化错误消息。
    """

    _error_color = 'red'

    def __new__(cls, error_message):
        """
        Creates a red-colored error message with the class name as the prefix. | 创建带有类名前缀的红色错误消息。

        Args:
            error_message (str): The error message to display. | 要显示的错误消息。

        Returns:
            str: The formatted error message. | 返回格式化的错误消息。
        """
        #Message.get_location(inspect.currentframe())
        return super(Error, cls).__new__(cls, text=error_message, color=cls._error_color, prefix=cls.__name__)


class Warning(Message):
    """
    A class for handling warning messages with a default yellow color. | 处理带有默认黄色的警告消息的类。

    Methods:
        __new__(cls, warning_message): Returns a formatted warning message with the class name as prefix. | 返回带有类名前缀的格式化警告消息。
    """

    _warning_color = 'yellow'
    
    def __new__(cls, warning_message):
        """
        Creates a yellow-colored warning message with the class name as the prefix. | 创建带有类名前缀的黄色警告消息。

        Args:
            warning_message (str): The warning message to display. | 要显示的警告消息。

        Returns:
            str: The formatted warning message. | 返回格式化的警告消息。
        """
        return super(Warning, cls).__new__(cls, text=warning_message, color=cls._warning_color, prefix=cls.__name__)
    

class Info(Message):
    """
    A class for handling informational messages with a default green color. | 处理带有默认绿色的消息类。

    Methods:
        __new__(cls, info_message): Returns a formatted informational message with the class name as prefix. | 返回带有类名前缀的格式化消息。
    """

    _info_color = 'green'
    
    def __new__(cls, info_message):
        """
        Creates a green-colored informational message with the class name as the prefix. | 创建带有类名前缀的绿色消息。

        Args:
            info_message (str): The informational message to display. | 要显示的消息。

        Returns:
            str: The formatted informational message. | 返回格式化的消息。
        """
        return super(Info, cls).__new__(cls, text=info_message, color=cls._info_color, prefix=cls.__name__)
    
class Progress(Message):
    """
    A class to create and manage a progress bar with adjustable refresh rate and length. | 创建和管理进度条的类，可调节刷新率和长度。

    Attributes:
        bar_base (str): The characters used to fill the progress bar. | 用于填充进度条的字符。
        refrsh_rate (float): The refresh rate of the progress bar in seconds. | 进度条的刷新率（以秒为单位）。
        bar_length (int): The length of the progress bar. | 进度条的长度。
        last_update (float): The time of the last bar update. | 上次进度条更新的时间。
        last_bar (str): The last displayed progress bar. | 上次显示的进度条。

    Methods:
        current_time(): Returns the current time in seconds. | 返回当前时间（秒）。
        refresh(): Checks if the bar should be refreshed based on refresh rate. | 根据刷新率检查是否需要刷新进度条。
        update_bar(bar): Updates the progress bar. | 更新进度条。
        progress_bar(current, total, text): Displays the progress bar based on the current and total values. | 根据当前和总值显示进度条。
    """

    def __new__(cls, **kwargs):
        """
        Initializes the Progress class and returns a new instance of it. | 初始化Progress类并返回其实例。

        Returns:
            Progress: A new instance of the Progress class. | Progress类的一个新实例。
        """
        super(Progress, cls).__new__(cls)
        return super(Color, cls).__new__(cls)

    def __init__(self, **kwargs):
        """
        Initializes the progress bar with optional parameters. | 使用可选参数初始化进度条。

        Args:
            refresh_rate (float, optional): The refresh rate of the progress bar. Default is 0.1 seconds. | 进度条的刷新率，默认0.1秒。
            bar_length (int, optional): The length of the progress bar. Default is 40. | 进度条的长度，默认40。
            show_time (bool, optional): Whether to show time cost. Default is False | 是否显示耗时，默认 False。
        """
        self._info_color = 'blue'
        self.bar_base = '█▓▒░' # 100% filled, 75% filled, 50% filled, 25% filled | 100% 填充，75% 填充，50% 填充，25% 填充
        self.refrsh_rate = kwargs.get('refresh_rate', 0.1) # By default, refresh bar every 0.1 seconds | 默认每0.1秒刷新一次
        self.bar_length = kwargs.get('bar_length', 40)
        self.show_time = kwargs.get('show_time', False)
        self.time_digit = kwargs.get('time_digit', 2)

        self.first_update = self.current_time()
        self.last_update = self.current_time()
        self.last_bar = ''

    def current_time(self,):
        '''
        return current time | 返回当前时间
        '''
        return time.time()

    def refresh(self,):
        '''
        whether to refresh bar | 是否刷新进度条
        '''
        return (self.current_time() - self.last_update) > self.refrsh_rate
    
    def update_bar(self, bar):
        """
        Updates the last progress bar and refreshes the update time. | 更新上次显示的进度条，并刷新更新时间。

        Args:
            bar (str): The updated progress bar string. | 更新后的进度条字符串。
        """
        self.last_update = self.current_time()
        self.last_bar = bar
    
    def progress_bar(self, current: int, total: int, text=None,):
        """
        Displays and refreshes the progress bar based on the current progress and total progress. | 根据当前进度和总进度显示并刷新进度条。

        Args:
            current (int): The current progress value. | 当前进度值。
            total (int): The total progress value. | 总进度值。
            text (str, optional): Additional text to display with the progress bar. | 可选的附加文本，与进度条一起显示。
        """
        fraction = current / total
        
        filled_length = int(self.bar_length * fraction)
        bar = self.bar_base[0] * filled_length + '-' * (self.bar_length - filled_length)
        
        percent = fraction * 100
        if self.refresh() or current == total:
            bar_info = f'{UP_ONE_LINE}\r{text}\n[{current}/{total}]|{bar}| {percent:.2f}%'
            if self.show_time:
                time_cost = round(self.last_update - self.first_update, self.time_digit)
                time_info = f'{time_cost}s'
                bar_info = f'{bar_info} | {time_info}'
            print(bar_info, end='',)
            self.update_bar(bar_info)
        else:
            print(self.last_bar, end='',)

        if current == total:
            print()