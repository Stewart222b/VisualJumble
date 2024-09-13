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
    @classmethod
    def class_name(cls):
        return cls.__name__

    def __new__(cls, text, **kwargs):
        '''
        args:
            text
        kwargs:
            color
            prefix
            ...
        '''

        color = kwargs.get('color', 'red')
        prefix = kwargs.get('prefix', None)

        if prefix:
            message = f'[{prefix}] {text}'
        else:
            message = f'{text}'
        
        return Color(message, color)

class Error(Message):
    _error_color = 'red'

    def __new__(cls, error_message):
        return super(Error, cls).__new__(cls, error_message, color=cls._error_color, prefix=cls.__name__)


class Warning(Message):
    _warning_color = 'yellow'
    
    def __new__(cls, warning_message):
        return super(Warning, cls).__new__(cls, warning_message, color=cls._warning_color, prefix=cls.__name__)
    

class Info(Message):
    _warning_color = 'green'
    
    def __new__(cls, warning_message):
        return super(Warning, cls).__new__(cls, warning_message, color=cls._warning_color, prefix=cls.__name__)
    