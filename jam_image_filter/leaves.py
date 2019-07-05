import os
import sys
import math
from PIL import Image, ImageOps
import util
import colorsys
import random

def leaves(image):
    image = image.convert('RGB')
    colours = util.get_dominant_colours(image, 8)
    light, dark = [colours[i] for i in random.sample(range(len(colours)), 2)]

    layer = Image.open(os.path.dirname(os.path.abspath(__file__)) + '/' +
                       'assets/leaves.png')

    layer = layer.convert('RGB')
    layer = ImageOps.grayscale(layer)
    layer = ImageOps.colorize(layer, dark, light)

    width, height = layer.size

    # shift half of the image horizontally
    half_width = width // 2

    left = layer.crop((0, 0, half_width, height))
    right = layer.crop((half_width, 0, width, height))
    new = Image.new('RGB', (width, height))
    new.paste(left, (half_width, 0))
    new.paste(right, (0, 0))

    return new

if __name__ == '__main__':
    im = Image.open(sys.argv[1])
    im = leaves(im)
    im.save(sys.argv[2], quality=90)
