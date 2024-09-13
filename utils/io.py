_RED = '\033[31m'
_GREEN = '\033[32m'
_YELLOW = '\033[33m'
_BLUE = '\033[34m'
_UNSET = '\033[0m'

class Color:
    _COLORS = {
        'red': _RED,
        'green': _GREEN,
        'yellow': _YELLOW,
        'blue': _BLUE,
    }

    def __new__(cls, text, color='red'):
        color_code = cls._COLORS.get(color, _UNSET)
        return f'{color_code}{text}{_UNSET}'
