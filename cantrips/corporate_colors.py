class Color:
    def __init__(self, hex_code: str):
        self.hex = hex_code

    @property
    def rgb(self):
        # Convert hex code to RGB tuple
        return tuple(int(self.hex[i : i + 2], 16) for i in (1, 3, 5))


# Define colors as instances of Color
BLUE = Color("#1E64C8")
YELLOW = Color("#FFD200")
WHITE = Color("#FFFFFF")
BLACK = Color("#000000")
ORANGE = Color("#F1A42B")
RED = Color("#DC4E28")
AQUA = Color("#2D8CA8")
PINK = Color("#E85E71")
SKY = Color("#8BBEE8")
LIGHTGREEN = Color("#AEB050")
PURPLE = Color("#825491")
WARMORANGE = Color("#FB7E3A")
TURQUOISE = Color("#27ABAD")
LIGHTPURPLE = Color("#BE5190")
GREEN = Color("#71A860")

# Define lists for COLORS, PRIMARY_COLORS, and SECONDARY_COLORS
COLORS = [
    BLUE,
    YELLOW,
    ORANGE,
    RED,
    AQUA,
    PINK,
    SKY,
    LIGHTGREEN,
    PURPLE,
    WARMORANGE,
    TURQUOISE,
    LIGHTPURPLE,
    GREEN,
]

PRIMARY_COLORS = [BLUE, YELLOW]
SECONDARY_COLORS = [
    ORANGE,
    RED,
    AQUA,
    PINK,
    SKY,
    LIGHTGREEN,
    PURPLE,
    WARMORANGE,
    TURQUOISE,
    LIGHTPURPLE,
    GREEN,
]
