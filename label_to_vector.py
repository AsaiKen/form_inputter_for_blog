import collections
import json
import os
import random
import re
from typing import Dict, List, Any

import MeCab
import numpy as np
import pandas as pd
from gensim import models, corpora, matutils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from image_to_labels import get_labels_and_rects, get_xywh

REDUCTOR_PKL = "static/tmp_reductor.pkl"
LR_PKL = 'static/tmp_lr.pkl'
ET_PKL = 'static/tmp_et.pkl'
DICT_PKL = "static/tmp_dict.pkl"
SCALER_PKL = "static/tmp_scaler.pkl"

CATEGORY2TOKENS_LIST_JSON = "static/tmp_category2tokens_list.json"
CATEGORY_CSV = "static/category.csv"
MIN_SIZE = 30
NUM_TOPICS = 30

# https://shogo82148.github.io/blog/2015/12/20/mecab-in-python3-final/
mecab_tokenizer = MeCab.Tagger("")
mecab_tokenizer.parse('')


def tokenize(text: str) -> List[str]:
  tokens = []
  node = mecab_tokenizer.parseToNode(text)
  while node:
    feats = node.feature.split(",")
    surface = node.surface
    if feats[0] in ['名詞'] or surface in ["あり", "有り", "アリ", "なし", "無し", "ナシ"]:
      surface = re.sub(r"\d", "0", surface)
      tokens.append(surface)
    node = node.next
  # print(text, tokens)
  return tokens


def vec2dense(vec, num_terms):
  return list(matutils.corpus2dense([vec], num_terms=num_terms).T[0])


def sparse_to_dense(vec):
  ret = [e[1] for e in vec]
  return ret


def get_tokens(labels: List[Dict[str, Any]], rects: List[Dict[str, int]], i: int) -> List[str]:
  x, y, w, h = get_xywh(rects[i])

  tokens = []
  added_texts = []

  # 中と左横、直下のテキストを見る
  for cur_y in range(y, y + h, 10):
    for label in labels:
      x2, y2, w2, h2 = get_xywh(label)
      text = label["text"]
      if text in added_texts:
        continue
      if y2 <= cur_y < y2 + h2 and (x2 <= x < x2 + w2 or x <= x2 < x + w or x2 + w2 < x):
        tokens += tokenize(text)
        added_texts.append(text)
  if tokens:
    return tokens

  # 見つからないなら、上のテキストを見る
  for cur_y in range(y - 10, y - 51, -10):
    for label in labels:
      x2, y2, w2, h2 = get_xywh(label)
      text = label["text"]
      if text in added_texts:
        continue
      if y2 <= cur_y < y2 + h2:
        tokens += tokenize(text)
        added_texts.append(text)
  if tokens:
    return tokens

  # 見つからないなら、下のテキストを見る
  for cur_y in range(y + h + 10, y + h + 51, 10):
    for label in labels:
      x2, y2, w2, h2 = get_xywh(label)
      text = label["text"]
      if text in added_texts:
        continue
      if y2 <= cur_y < y2 + h2:
        tokens += tokenize(text)
        added_texts.append(text)
  if tokens:
    return tokens

  # 見つからないなら、もっと上のテキストを見る
  for cur_y in range(y - 51, y - 101, -10):
    for label in labels:
      x2, y2, w2, h2 = get_xywh(label)
      text = label["text"]
      if text in added_texts:
        continue
      if y2 <= cur_y < y2 + h2:
        tokens += tokenize(text)
        added_texts.append(text)
  if tokens:
    return tokens

  # 見つからないなら、もっと下のテキストを見る
  for cur_y in range(y + h + 51, y + h + 101, 10):
    for label in labels:
      x2, y2, w2, h2 = get_xywh(label)
      text = label["text"]
      if text in added_texts:
        continue
      if y2 <= cur_y < y2 + h2:
        tokens += tokenize(text)
        added_texts.append(text)
  if tokens:
    return tokens

  return tokens


def get_typetext_count_tokens(rects: List[Dict[str, int]], i: int) -> List[str]:
  rect = rects[i]
  x, y, w, h = get_xywh(rect)

  tokens = []
  # 左横にあるtype=textの数を数える
  left = 0
  for i2 in range(len(rects)):
    if i == i2:
      continue
    rect2 = rects[i2]
    x2, y2, w2, h2 = get_xywh(rect2)
    if (y2 <= y + h / 2 < y2 + h2 or y <= y2 + h2 / 2 < y + h) and x2 + w2 < x:
      left += 1
  if left > 0:
    tokens.append("__LEFT_COUNT_%d__" % left)

  # 右横にあるtype=textの数を数える
  right = 0
  for i2 in range(len(rects)):
    if i == i2:
      continue
    rect2 = rects[i2]
    x2, y2, w2, h2 = get_xywh(rect2)
    if (y2 <= y + h / 2 < y2 + h2 or y <= y2 + h2 / 2 < y + h) and x + w < x2:
      right += 1
  if right > 0:
    tokens.append("__RIGHT_COUNT_%d__" % right)

  return tokens


def get_category2tokens_list():
  if os.path.exists(CATEGORY2TOKENS_LIST_JSON):
    with open(CATEGORY2TOKENS_LIST_JSON) as f:
      return json.load(f)

  df = pd.read_csv(CATEGORY_CSV)
  gs = df.query("deleted == False and done == True").groupby(["page_id", "form_id"])
  x = []
  y = []
  category2tokens_list = collections.defaultdict(list)
  for g in gs:
    page_id = g[0][0]
    form_id = int(g[0][1])
    rows = g[1]
    typetext_ids = rows["typetext_id"].astype("int").tolist()

    jpg_path = "static/data/%s/%d.jpg" % (page_id, form_id)
    rects_json_path = "static/data/%s/%d.json" % (page_id, form_id)
    labels, rects = get_labels_and_rects(jpg_path, rects_json_path)

    for i in typetext_ids:
      tokens = get_tokens(labels, rects, i)
      tokens += get_typetext_count_tokens(rects, i)
      x.append(tokens)

      row = rows.query("typetext_id == %d" % i).iloc[0]
      category = row["value"]
      y.append(category)

      category2tokens_list[category].append(tokens)
      print(page_id, form_id, i, tokens, category)

  with open(CATEGORY2TOKENS_LIST_JSON, "w") as f:
    json.dump(category2tokens_list, f, indent=2)

  return category2tokens_list


def get_XY(category2tokens_list):
  too_few_categories = []
  # サンプルデータ内のカテゴリの出現数を同じにする
  for category in category2tokens_list:
    tokens_list = category2tokens_list[category]
    if len(tokens_list) >= MIN_SIZE:
      random.shuffle(tokens_list)
      category2tokens_list[category] = tokens_list[:MIN_SIZE]
    else:
      too_few_categories.append(category)

  # 出現回数が少なすぎるカテゴリを除去
  for category in too_few_categories:
    del category2tokens_list[category]

  X = []
  Y = []
  for category in category2tokens_list:
    for tokens in category2tokens_list[category]:
      X.append(tokens)
      Y.append(category)

  print("%d unique labels" % len(set(Y)))
  X = np.array(X)
  Y = np.array(Y)

  # 2:1の比率で訓練データとテストデータに分割する
  kf = StratifiedKFold(n_splits=3, shuffle=True)
  X_train = Y_train = X_test = Y_test = None
  for train_index, test_index in kf.split(X, Y):
    print(len(train_index), len(test_index))
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]
    break

  # BoW化
  dictionary = corpora.Dictionary(X_train)
  corpus = [dictionary.doc2bow(x) for x in X_train]

  # 次元削減
  reductor = models.LsiModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
  X_train2, Y_train2 = apply_reductor(X_train, Y_train, dictionary, reductor)

  # 標準化
  scaler = StandardScaler()
  scaler.fit(X_train2)
  X_train2 = scaler.transform(X_train2)

  # テストデータも同様の処理をする
  X_test2, Y_test2 = apply_reductor(X_test, Y_test, dictionary, reductor)
  X_test2 = scaler.transform(X_test2)

  return X_train2, Y_train2, X_test2, Y_test2


def apply_reductor(X, Y, dictionary, reductor):
  X_reducted = []
  Y_reducted = []
  for i in range(len(X)):
    x = X[i]
    vec_bow = dictionary.doc2bow(x)
    vec_reducted = sparse_to_dense(reductor[vec_bow])
    # 全部0だと[]を返すので[0,0,...]に直す
    if len(vec_reducted) < NUM_TOPICS:
      vec_reducted = [0 for _ in range(NUM_TOPICS)]
    X_reducted.append(vec_reducted)
    Y_reducted.append(Y[i])
  return X_reducted, Y_reducted


if __name__ == "__main__":
  category2tokens_list_ = get_category2tokens_list()
  X_train_, Y_train_, X_test_, Y_test_ = get_XY(category2tokens_list_)
  print(X_train_, Y_train_, X_test_, Y_test_)
  pass
