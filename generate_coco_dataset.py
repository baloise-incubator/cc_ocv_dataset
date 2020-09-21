#!/usr/bin/env python3

from PIL import Image
from PIL import ImageDraw
import math
from numpy import matrix
from numpy import linalg
import requests
import os
import random


def download_images(imglist):
    d = os.path.dirname(imglist)
    ret = []
    idx = 0
    for line in open(imglist, "r").readlines():
        url = line.strip()
        if url:
            fn = os.path.join(d, str(idx) + ".jpg")
            ret.append(fn)
            idx += 1
            if os.path.exists(fn):
                print("Found", url)
                continue
            print("Downloading", url)
            r = requests.get(url)
            with open(fn, "wb") as f:
                f.write(r.content)
    return ret


class ImageGen:
    BG_SIZE = (1024, 720)
    im = None
    bg = None

    def __init__(self, image, background):
        self.im = Image.open(image).convert("RGBA")
        self.bg = Image.open(background)
        scale = min(self.bg.size[0] / self.BG_SIZE[0], self.bg.size[1] / self.BG_SIZE[1])
        self.bg = self.bg.resize(self.BG_SIZE)

    def add_corners_to_img(self, rad):
        circle = Image.new("L", (rad * 2, rad * 2), 0)
        draw = ImageDraw.Draw(circle)
        draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
        alpha = Image.new("L", self.im.size, "white")
        w, h = self.im.size
        alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
        alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
        alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
        alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
        self.im.putalpha(alpha)

    @staticmethod
    def find_coeffs(pa, pb):
        import numpy

        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = numpy.matrix(matrix, dtype=numpy.float)
        B = numpy.array(pb).reshape(8)

        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)

    def gen_matrix(self, size_x_top, size_x_bottom, size_y_left, size_y_right):
        w, h = self.im.size
        h_left_px2 = (h * (1 - size_y_left)) / 2
        h_right_px2 = (h * (1 - size_y_right)) / 2
        w_top_px2 = (w * (1 - size_x_top)) / 2
        w_bottom_px2 = (w * (1 - size_x_bottom)) / 2
        orig = [[0, 0], [w, 0], [0, h], [w, h]]
        new = [
            [w_top_px2, h_left_px2],
            [w - w_top_px2, h_right_px2],
            [w_bottom_px2, h - h_left_px2],
            [w - w_bottom_px2, h - h_right_px2],
        ]
        return self.find_coeffs(new, orig)

    def generate(self, result_image):
        width, height = self.im.size
        # crop image to elimimnate unclear corners
        self.im = self.im.crop((5, 5, width - 10, height - 10))
        width, height = self.im.size
        # cleanup corners
        self.add_corners_to_img(20)
        # add rotation
        self.im = self.im.rotate(random.randint(-20, 20), expand=True)

        # random values for perspective transformation:
        r1 = random.randint(80, 100) / 100
        r2 = random.randint(80, 100) / 100
        r3 = random.randint(80, 100) / 100
        r4 = random.randint(80, 100) / 100
        self.im = self.im.transform(
            self.im.size, Image.PERSPECTIVE, self.gen_matrix(r1, r2, r3, r4), Image.BICUBIC
        )

        width = width + 200
        height = height + 100
        new_img = Image.new("RGBA", (width, height))
        new_img.paste(self.bg, (0, 0))
        new_img.paste(self.im, (10, 10), mask=self.im.split()[3])
        new_img.save(result_image)


if __name__ == "__main__":
    bg_lst = download_images("background/images.txt")
    id_lst = download_images("idcards/images.txt")
    print("idcards =", len(id_lst), ", bgimages =", len(bg_lst))
    idx = 0
    for bg_img in bg_lst:
        for id_img in id_lst:
            imggen = ImageGen(id_img, bg_img)
            imggen.generate(os.path.join("data", str(idx) + ".png"))
            idx += 1
