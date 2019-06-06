import json
import math
import os
import re
from typing import Dict, List, Any

import cv2
import imutils
import numpy as np
import pyocr.builders
import regex
from PIL import Image, ImageFont, ImageDraw

FONT = ImageFont.truetype('./ipagp.ttf', 20)

PYOCR_TOOL = pyocr.get_available_tools()[0]
PYOCR_LANGS = PYOCR_TOOL.get_available_languages()
assert "Japanese" in PYOCR_LANGS

KERNEL_SQ_LARGE = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
KERNEL_SQ_SMALL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
KERNEL_RECT_SMALL = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))


def get_labels(jpg_path: str) -> List[Dict[str, Any]]:
  print(jpg_path)
  assert os.path.exists(jpg_path), jpg_path

  labels_path = jpg_path + ".labels"
  if os.path.exists(labels_path):
    with open(labels_path) as f:
      return json.load(f)

  result = []

  image = cv2.imread(jpg_path)
  img_h, img_w, _ = image.shape

  # グレースケール
  gray = image.copy()
  gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

  hat = gray.copy()
  if np.mean(hat) > 127:
    # 白地に黒字 -> 黒を強調
    hat = cv2.morphologyEx(hat, cv2.MORPH_BLACKHAT, KERNEL_SQ_LARGE)
  else:
    # 黒字に白字 -> 白を強調
    hat = cv2.morphologyEx(hat, cv2.MORPH_TOPHAT, KERNEL_SQ_LARGE)

  # 文字を膨張させる
  dilate = hat.copy()
  dilate = cv2.morphologyEx(dilate, cv2.MORPH_DILATE, KERNEL_SQ_SMALL)

  # 縦方向の勾配を抽出
  grad_x = dilate.copy()
  grad_x = cv2.Sobel(grad_x, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
  grad_x = np.absolute(grad_x)
  (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
  if min_val < max_val:
    grad_x = (255 * ((grad_x - min_val) / (max_val - min_val)))
  grad_x = grad_x.astype("uint8")

  # 縦線を黒塗りする。ただし、確率的なのであまり信用しない。
  fill_line = grad_x.copy()
  lines = cv2.HoughLinesP(fill_line, 1, np.pi / 2, 10, minLineLength=50, maxLineGap=0)
  long_lines = cv2.HoughLinesP(fill_line, 1, np.pi / 2, 10, minLineLength=100, maxLineGap=0)
  if lines is not None:
    for line in lines:
      for x1, y1, x2, y2 in line:
        cv2.line(fill_line, (x1, y1), (x2, y2), 0, 3)

  # 勾配領域を膨張させる
  dilate2 = fill_line.copy()
  dilate2 = cv2.morphologyEx(dilate2, cv2.MORPH_DILATE, KERNEL_SQ_LARGE)

  base_img = get_base_img(dilate2.copy())
  sub_img = get_sub_img(dilate2.copy(), base_img)
  new_img = base_img | sub_img

  # 2つのimgをなじませる
  dilate3 = new_img.copy()
  dilate3 = cv2.morphologyEx(dilate3, cv2.MORPH_DILATE, KERNEL_RECT_SMALL)

  # 縦線を黒塗りし直す
  fill_line2 = dilate3.copy()
  if long_lines is not None:
    for line in long_lines:
      for x1, y1, x2, y2 in line:
        cv2.line(fill_line2, (x1, y1), (x2, y2), 0, 3)

  cnts = cv2.findContours(fill_line2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  pos2crop = {}
  cropid2pos = {}
  total_height = 0
  max_width = 0
  for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    if w < 30 or h < 30 or w < h:
      continue
    # 範囲を切り取り、2値化する
    crop = gray[y:y + h, x:x + w].copy()
    crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # tesseractの精度を上げるために、白背景＋黒文字にする
    if np.mean(crop) <= 127:
      crop = ~crop
    assert crop.shape == (h, w)
    pos = (x, y, w, h)
    pos2crop[pos] = crop
    cropid2pos[id(crop)] = pos
    total_height += math.ceil(h * 1.1)
    max_width = max(w, max_width)

  if total_height == 0 or max_width == 0:
    return result

  ocr_img = np.zeros((total_height, max_width)) + 255
  cur_y = 0
  range2crop = {}
  for (x, y, w, h) in sorted(pos2crop):
    crop = pos2crop[(x, y, w, h)]
    assert crop.shape == (h, w)
    assert ocr_img[cur_y:cur_y + h, 0:w].shape == (h, w)
    ocr_img[cur_y:cur_y + h, 0:w] = crop
    range2crop[(cur_y, cur_y + h)] = crop
    cur_y += math.ceil(h * 1.1)

  assert cur_y == total_height

  # OCR
  image_open = Image.fromarray(np.uint8(ocr_img))
  ocr_results = PYOCR_TOOL.image_to_string(image_open, lang="Japanese",
                                           builder=pyocr.builders.LineBoxBuilder(tesseract_layout=6))
  pos2content = {}
  for ocr_res in ocr_results:
    content = ocr_res.content
    y = ocr_res.position[0][1]
    crop = None
    for y_start, y_end in range2crop:
      if y_start <= y < y_end:
        crop = range2crop[(y_start, y_end)]
        break
    if crop is None:
      # print("[!] crop not found: %s" % ocr_res)
      continue

    pos = cropid2pos[id(crop)]
    if pos is None:
      # print("[!] pos not found: %s" % ocr_res)
      continue

    content = re.sub(r'\s+', '', content)
    if len(content) >= 2 or (len(content) == 1 and regex.match(r'\p{Han}', content[0])):
      # puppeteerのscreenshotは2倍のサイズで作成されるので、フォームのサイズに戻す
      pos = (pos[0] / 2, pos[1] / 2, pos[2] / 2, pos[3] / 2)
      pos2content.setdefault(pos, "")
      pos2content[pos] += content

  for pos in pos2content:
    x, y, width, height = pos
    result.append({"x": x, "y": y, "width": width, "height": height, "text": pos2content[pos]})

  with open(labels_path, "w") as f:
    json.dump(result, f, indent=2)
  return result


def get_base_img(img: np.ndarray) -> np.ndarray:
  thresh = img
  thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  return thresh


def get_sub_img(img: np.ndarray, base_img: np.ndarray) -> np.ndarray:
  thresh = img
  thresh[base_img == 255] = 0

  thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, KERNEL_SQ_SMALL)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, KERNEL_SQ_LARGE)

  # ゴミを削除
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    if w < 50 or (h < 50 and w < 150) or h / w > 1:
      cv2.rectangle(thresh, (x, y), (x + w, y + h), 0, cv2.FILLED)

  return thresh


def modify_labels(labels: List[Dict[str, Any]]):
  for label in labels:
    pattern = r"[\s!\"#$%&()*+,./:;<=>?\[\\\]^_`{|}　、。，．・：；？！´｀¨＾￣＿〃仝〆／＼～∥｜…‥‘“”（）〔〕［］｛｝〈〉《》「」『』【】]+"
    label["text"] = re.sub(pattern, "、", label["text"])
  pass


def get_labels_and_rects(jpg_path: str, rects_json_path: str):
  labels = get_labels(jpg_path)
  modify_labels(labels)
  with open(rects_json_path) as f:
    rects = json.load(f)
  assert type(rects) == list
  return labels, rects


def image_to_labels(jpg_path: str, rects_json_path: str):
  labels, rects = get_labels_and_rects(jpg_path, rects_json_path)
  save_debug_jpg(jpg_path)
  return labels, rects


def save_debug_jpg(jpg):
  debug_jpg = "%s_debug.jpg" % (jpg.replace(".jpg", ""))
  if os.path.exists(debug_jpg):
    return
  labels = get_labels(jpg)
  debug_img = cv2.imread(jpg)
  for label in labels:
    x, y, w, h = get_xywh(label)
    x, y, w, h = x * 2, y * 2, w * 2, h * 2
    debug_img = put_label(debug_img, x, y, label["text"])
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
  cv2.imwrite(debug_jpg, debug_img)


def get_xywh(rect: Dict[str, Any]):
  return int(rect["x"]), int(rect["y"]), int(rect["width"]), int(rect["height"])


def put_label(image, x, y, label):
  img_pil = Image.fromarray(image)
  draw = ImageDraw.Draw(img_pil)
  draw.text((x, y - 20), label, font=FONT, fill=(0, 0, 255))
  image = np.array(img_pil)
  return image


if __name__ == "__main__":
  labels_, rects_ = image_to_labels("static/data/2fafe42e/1.jpg", "static/data/2fafe42e/1.json")
  print(labels_)
  print(rects_)
  pass
