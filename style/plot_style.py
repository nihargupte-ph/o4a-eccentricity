import os
import matplotlib.pyplot as plt


def initialize_latex():
    # usetex / minus handling now lives in paper.mplstyle (see the Fonts block
    # there): the PDF backend + usetex drops the math-minus glyph, so we render
    # via mathtext instead. Applying the style is all that's needed.
    style_path = os.path.join(os.path.dirname(__file__), "paper.mplstyle")
    plt.style.use(style_path)


def tex_event(name):
    """Format an event name (e.g. 'GW230706_104333') for a mathtext label.

    Under mathtext + cmr10 a plain-text underscore maps to the font's dotaccent
    glyph (an OT1-encoding quirk), so a raw name renders as 'GW230706<dot>104333'.
    Wrapping the name in upright math (\\mathrm) with an escaped underscore makes
    it render as a real underscore, identically to the old usetex output.
    """
    return r"$\mathrm{" + str(name).replace("_", r"\_") + "}$"


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
