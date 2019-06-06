会社ブログ用に作ったフォーム要素の入力値を予想するAI

インストール
---

Mac mojave

```
$ brew install mecab mecab-ipadic swig tesseract
$ wget https://github.com/tesseract-ocr/tessdata_best/raw/master/script/Japanese.traineddata -P /usr/local/Cellar/tesseract/4.0.0_1/share/tessdata/
$ wget https://github.com/tesseract-ocr/tessdata_best/raw/master/script/Japanese_vert.traineddata -P /usr/local/Cellar/tesseract/4.0.0_1/share/tessdata/
$ wget https://github.com/tesseract-ocr/tessdata_best/raw/master/jpn.traineddata -P /usr/local/Cellar/tesseract/4.0.0_1/share/tessdata/
$ wget https://github.com/tesseract-ocr/tessdata_best/raw/master/jpn_vert.traineddata -P /usr/local/Cellar/tesseract/4.0.0_1/share/tessdata/
$ brew install cmake opencv@3
$ brew link --force opencv@3
$ pip3 install --upgrade pip
$ pip3 install -r requirements.txt
```
