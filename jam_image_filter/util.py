import math
import colorsys
import scipy
import scipy.cluster
import operator
import math
from PIL import Image
import numpy as np
import random

WIDTH = 1700
HEIGHT = 540

def rgb_to_gray(r, g, b, a = None):
    return 0.299 * r + 0.587 * g + 0.114 * b

def get_avg_gray(pix, x, y, radius, sample_size = 0.1):
    nsamples = math.pow(radius, 2) * sample_size
    avg = 0
    for y in range(y - radius, y + radius, int(1 / sample_size)):
        for x in range(x - radius, x + radius, int(1 / sample_size)):
            try:
                if len(pix[x, y]) >= 3:
                    avg += rgb_to_gray(*pix[x,y])
                else:
                    avg += pix[x, y]
            except IndexError:
                pass
    return 1 - avg / nsamples / 255.0

def get_dominant_colours(im, n):
    small_width = 50
    small_height = 50
    orig_width, orig_height = im.size
    im = im.resize((small_width, small_height))

    # Was
    #
    #   array = scipy.misc.fromimage(im)
    #
    # N.B. scipy.cluster.vq.kmeans requires floats or doubles.
    array = np.asarray(im, dtype=np.float32)
    array = array.reshape(small_width * small_height, array.shape[2])

    # array is now a list of 250 tuples, e.g. 3-tuples for the RGB colour model.

    codes, dist = scipy.cluster.vq.kmeans(array, n)
    codes = pad_colours(codes, n)

    #vec, dist = scipy.cluster.vq.vq(array, codes)
    #vec = vec.reshape(width, height)
    #scale = (orig_width / float(small_width), orig_height / float(small_height))
    return codes#, vec, scale

def pad_colours(rgbs, n):
    new_rgbs = [None] * n
    for i in range(n):
        j = int(i / (n / float(len(rgbs))))
        new_rgbs[i] = rgbs[j]
    return new_rgbs


def _rgb_to_hue(rgb):
    # colorsys requires that all coordinates are between 0 and 1.
    t = rgb / 255.0

    return colorsys.rgb_to_hls(*t)[0]

def order_colours_by_hue(colours):
    return sorted(colours, key=_rgb_to_hue, reverse=True)


# Should probably be called _rgb_to_luminosity...
def _rgb_to_brightness(rgb):
    t = rgb / 255.0

    return colorsys.rgb_to_hls(*t)[1]

def order_colours_by_brightness(colours):
    return sorted(colours, key=_rgb_to_brightness, reverse=True)


def _rgb_to_saturation(rgb):
    t = rgb / 255.0

    return colorsys.rgb_to_hls(*t)[2]

def order_colours_by_saturation(colours):
    return sorted(colours, key=_rgb_to_saturation, reverse=True)


def resize_jam_background(im, target_width=WIDTH, target_height=HEIGHT,
                          max_resize=16.0, pixelated=False):
    width, height = im.size
    scale, crop_left, crop_right, crop_top, crop_bottom = \
        get_resize_params(im, target_width, target_height, max_resize)

    if pixelated:
        im = im.resize((int(width * scale), int(height * scale)))
    else:
        im = im.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    im = im.crop((crop_left, crop_top, crop_right, crop_bottom))

    return im

def centre_crop(im, width, height):
    im_width, im_height = im.size
    width = min(width, im_width)
    height = min(height, im_height)
    left = max(0, (im_width - width) / 2)
    top = max(0, (im_height - height) / 2)

    return im.crop((left, top, left + width, top + height))

def random_crop(im, width, height):
    im_width, im_height = im.size
    width = min(width, im_width)
    height = min(height, im_height)

    if im_width > width:
        left = random.randint(0, im_width - width - 1)
    else:
        left = 0
    if im_height > height:
        top = random.randint(0, im_height - height - 1)
    else:
        top = 0

    return im.crop((left, top, left + width, top + height))

def get_resize_params(im, target_width=WIDTH, target_height=HEIGHT, max_resize=4.0):
    width, height = im.size

    scale = min(max_resize, max(target_width / float(width),
                                target_height / float(height)))
    width = int(width * scale)
    height = int(height * scale)

    crop_left = int(max(0, (width - target_width) / 2))
    crop_right = int(width - crop_left)
    crop_top = int(max(0, (height - target_height) / 2))
    crop_bottom = int(height - crop_top)
    
    return scale, crop_left, crop_right, crop_top, crop_bottom

def max_size(im, target_width, target_height):
    width, height = im.size
    scale = max(width / target_width, height / target_height)
    if scale > 1:
        return im.resize((int(width / scale), int(height / scale)))
    else:
        return im

def interpolate_colour(colours, x):
    i = (len(colours) - 1) * x
    start_i = int(math.floor(i))
    end_i = start_i + 1
    delta = i - start_i

    start_rgb = map(int, colours[start_i])
    end_rgb = map(int, colours[end_i])

    rgb = []
    for i in range(3):
        rgb.append(int(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * delta))

    return tuple(rgb)

def create_gradient(size, colours):
    vertical = True # hard coded for now
    width, height = size
    im = Image.new('RGB', (1, height))
    pix = im.load()

    for i in range(height):
        pix[0, i] = interpolate_colour(colours, i / float(height))

    im = im.resize((width, height), Image.BILINEAR)
    return im

# rgb = (r, g, b), where 0 <= r, g, b <= 255
# returns (h, l, s), where 0 <= h, l, s <= 255
def rgb_to_hls(rgb):
    rgb = map(lambda x: x / 255.0, rgb)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return tuple(map(lambda x: x * 255.0, [h, l, s]))

def hls_to_rgb(hls):
    hls = map(lambda x: x / 255.0, hls)
    r, g, b = colorsys.hls_to_rgb(*hls)
    return tuple(map(lambda x: int(x * 255.0), [r, g, b]))

def modify_hls(rgb, h=None, l=None, s=None):
    hls = list(rgb_to_hls(rgb))
    mods = [h, l, s]
    for i, mod in enumerate(mods):
        if mod:
            old = hls[i]
            hls[i] = max(min(mod(hls[i]), 255), 0)
    return hls_to_rgb(hls)

def rgb_to_hsv(rgb):
    rgb = map(lambda x: x / 255.0, rgb)
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return tuple(map(lambda x: x * 255.0, [h, s, v]))

def hsv_to_rgb(hsv):
    hsv = map(lambda x: x / 255.0, hsv)
    r, g, b = colorsys.hsv_to_rgb(*hsv)
    return tuple(map(lambda x: int(x * 255.0), [r, g, b]))

def modify_hsv(rgb, h=None, s=None, v=None):
    hsv = list(rgb_to_hsv(rgb))
    mods = [h, s, v]
    for i, mod in enumerate(mods):
        if mod:
            old = hsv[i]
            hsv[i] = max(min(mod(hsv[i]), 255), 0)
    return hsv_to_rgb(hsv)

def spread_colours_by_lightness(rgbs, min_lightness=50, max_lightness=220):
    lightnesses = sorted(map(lambda rgb: rgb_to_hls(rgb)[1], rgbs))
    bottom = lightnesses[0]
    top = lightnesses[-1]
    scale = float((max_lightness - min_lightness)) / (top - bottom)

    def modify_lightness(l):
        l = l - bottom
        l = l * scale
        l = l + min_lightness
        return l

    for i, rgb in enumerate(rgbs):
        rgbs[i] = modify_hls(rgb, l=modify_lightness)

    return rgbs
