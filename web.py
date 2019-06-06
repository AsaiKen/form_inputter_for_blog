import json
import json

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


class Page():
  def __init__(self, page_id, form_id, typetext_id, candidates, form_rect, tt_rect, v):
    self.page_id = page_id
    self.form_id = form_id
    self.typetext_id = typetext_id
    self.candidates = candidates
    self.form_rect = form_rect
    self.tt_rect = tt_rect
    self.v = v
    pass


@app.route("/", methods=['POST', 'GET'])
def index():
  df = pd.read_csv("static/category.csv")
  print(df.shape)

  if request.method == 'POST':
    print("POST")
    page_id = request.form["page_id"]
    form_id = int(request.form["form_id"])
    typetext_id = int(request.form["typetext_id"])
    v = request.form["v"]
    print((page_id, form_id, typetext_id, v))
    if "delete" in request.form:
      print("delete")
      row = df.query("page_id == '%s' and form_id == %d and typetext_id == %d" % (page_id, form_id, typetext_id))
      print(row.index)
      df.loc[row.index, "deleted"] = True
      df.loc[row.index, "done"] = True
    elif "submit" in request.form:
      print("submit")
      row = df.query("page_id == '%s' and form_id == %d and typetext_id == %d" % (page_id, form_id, typetext_id))
      print(row.index)
      df.loc[row.index, "value"] = v
      df.loc[row.index, "done"] = True
    else:
      print("error")
      raise Exception
    df.to_csv("category.csv", index=False)

  rows = df.query("done == False")
  if len(rows) == 0:
    return render_template('index.html')

  row = rows.iloc[0]
  page_id = row["page_id"]
  form_id = int(row["form_id"])
  typetext_id = int(row["typetext_id"])
  v = "テスト{{任意}}"  # row["value"]
  print((page_id, form_id, typetext_id, v))
  candidates = sorted(df["value"].astype("str").unique().tolist())
  with open("static/data/%s/%d_texts.json" % (page_id, form_id)) as f:
    d = json.load(f)
  form_rect = d["form"]["rect"]
  tt_rect = d["typeTexts"][typetext_id]["rect"]
  page = Page(page_id, form_id, typetext_id, candidates, form_rect, tt_rect, v)

  return render_template('index.html', page=page)


if __name__ == "__main__":
  app.run()
