import os
import matplotlib.pyplot as plt


def initialize_latex():
    plt.rc("text", usetex=True)
    style_path = os.path.join(os.path.dirname(__file__), "..", "style", "test.mplstyle")
    plt.style.use(style_path)


COLORS = {
    "dark purple": "#524c78",
    "purple": "#6a51a3",
    "grey": "#8C8C8C",
    "lavender": "#8882ae",
    "green": "#55A868",
    "orange": "#DD8452",
    "blue": "#4C72B0",
    "pink": "#DA8BC3",
    "yellow": "#FFCC00",
    "red": "#E24A33",
    "brown": "#8B4513",
    "mint": "#88CC88",
    "maroon": "#650B2C",
}
