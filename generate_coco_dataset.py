#!/usr/bin/env python3

from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
import math
import numpy as np
import requests
import os
import random
import json
from datetime import date
from skimage import measure  # (pip install scikit-image)
from skimage import io
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)


def download_images(imglist):
    d = os.path.dirname(imglist)
    ret = []
    
    for line in open(imglist, "r").readlines():
        url = line.strip()
        if url:
            fn = os.path.join(d, os.path.basename(url))
            ret.append(fn)
            if os.path.exists(fn):
                print("Found", url)
                continue
            print("Downloading", url)
            r = requests.get(url)
            with open(fn, "wb") as f:
                f.write(r.content)
    return ret


def add_corners_to_img(im, rad):
    circle = Image.new("L", (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new("L", im.size, "white")
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def gen_matrix(im, size_x_top, size_x_bottom, size_y_left, size_y_right):
    w, h = im.size
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
    return find_coeffs(new, orig)


# c.f. https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#previewing-coco-annotations
def create_sub_mask_annotation(
    sub_mask, image_id, category_id, annotation_id, is_crowd
):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation="low")

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentation = [round(x, 3) for x in segmentation]
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        "segmentation": segmentations,
        "iscrowd": is_crowd,
        "image_id": image_id,
        "category_id": category_id,
        "id": annotation_id,
        "bbox": bbox,
        "area": area,
    }

    return annotation


class ImageGen:
    BG_SIZE = (1024, 720)

    def __init__(self, image, background):
        self.r1 = random.randint(80, 100) / 100
        self.r2 = random.randint(80, 100) / 100
        self.r3 = random.randint(80, 100) / 100
        self.r4 = random.randint(80, 100) / 100
        self.r5 = random.randint(-15, 15)  # rotation
        self.im = Image.open(image)
        im_size = (self.BG_SIZE[0] - 200, self.BG_SIZE[1] - 200)
        # crop image to elimimnate unclear corners
        width, height = self.im.size
        self.im = self.im.crop((5, 5, width - 10, height - 10))
        self.im = self.im.resize(im_size).convert("RGBA")
        self.bg = Image.open(background)
        self.bg = self.bg.resize(self.BG_SIZE)

    def _generate_image(self, im, bg):
        # cleanup corners
        add_corners_to_img(im, 20)

        # random values for perspective transformation:
        im = im.transform(
            im.size,
            Image.PERSPECTIVE,
            gen_matrix(im, self.r1, self.r2, self.r3, self.r4),
            Image.BICUBIC,
        )

        # add rotation
        im = im.rotate(self.r5, expand=True)
        new_img = bg.copy()
        new_img.paste(im, (20, 20), mask=im.split()[3])
        return new_img.convert("RGB")

    def generate_image(self, result_image):
        img = self._generate_image(self.im, self.bg)
        img.save(result_image)

    def generate_mask(self, result_image):
        # generate mask for polygon extraction
        bg = Image.new("L", self.BG_SIZE, 0)
        mask = Image.new("L", self.im.size, 255).convert("RGBA")
        img = self._generate_image(mask, bg)
        img.convert("1").save(result_image)

    def gen_annotation(self, result_image, image_id, annotation_id):
        img = io.imread(result_image)
        # polygon extraction
        return create_sub_mask_annotation(img, image_id, annotation_id, 0, False)


if __name__ == "__main__":
    bg_lst = download_images("background/images.txt")
    card_lst = download_images("idcards/images.txt")
    print("idcards =", len(card_lst), ", bgimages =", len(bg_lst))

    annotation_id = 1
    idx = 0
    images = list()
    annotations = list()
    for bg_img in bg_lst:
        for card_img in card_lst:
            image_path = os.path.join("images", str(idx) + ".jpg")
            mask_path = os.path.join("images", str(idx) + "_mask.png")
            imggen = ImageGen(card_img, bg_img)
            imggen.generate_image(image_path)
            imggen.generate_mask(mask_path)

            images.append(
                {
                    "width": imggen.BG_SIZE[0],
                    "height": imggen.BG_SIZE[1],
                    "id": idx,
                    "file_name": image_path,
                }
            )

            annotation = imggen.gen_annotation(mask_path, idx, annotation_id)
            annotations.append(annotation)
            idx += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"supercategory": "card", "id": annotation_id, "name": "idcard"}
        ],
        "info": {
            "description": "ID Card Dataset",
            "url": "https://github.com/baloise-incubator/cc_ocv_dataset",
            "version": "1.0",
            "year": 2020,
            "contributor": "Baloise",
            "date_created": date.today().strftime("%Y/%m/%d"),
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            }
        ],
    }

    with open("annotations.json", "w") as fp:
        json.dump(coco, fp, indent=2)
