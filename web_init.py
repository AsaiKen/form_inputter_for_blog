import glob
import json
import pandas as pd
import os


def create_resultcsv():
  txt = "static/category.csv"
  if os.path.exists(txt):
    print("%s exists" % txt)
    return

  with open(txt, "w") as out_f:
    out_f.write("page_id,form_id,typetext_id,value,deleted,done\n")
    jpgs = glob.glob("static/data/**/*.jpg")
    for jpg in sorted(jpgs):
      page_id, form_id = jpg.replace("static/data/", "").replace(".jpg", "").split("/")
      form_id = int(form_id)
      # print(page_id, form_id)
      with open(jpg.replace(".jpg", "_texts.json")) as f:
        d = json.load(f)
      assert type(d) == dict
      assert type(d["typeTexts"]) == list
      for k in d["typeTexts"]:
        typetext_id = k["index"]
        print(page_id, form_id, typetext_id)
        out_f.write("%s,%d,%d,%s,%s,%s\n" % (page_id, form_id, typetext_id, "{{未設定}}", False, False))

  assert pd.read_csv(txt) is not None


def translate_format():
  names = glob.glob("static/data/*/*_texts.json")
  print(names)
  for name in names:
    rects = []
    with open(name) as f:
      d = json.load(f)
    type_texts = d["typeTexts"]
    form_rect = d["form"]["rect"]
    for tt in type_texts:
      # print(tt["rect"])
      rect = tt["rect"]
      rect["x"] -= form_rect["x"]
      rect["y"] -= form_rect["y"]
      rects.append(rect)
    print(rects)
    with open(name.replace("_texts.json", ".json"), "w") as f:
      json.dump(rects, f, indent=2)
    os.remove(name)


if __name__ == "__main__":
  # create_resultcsv()
  translate_format()
