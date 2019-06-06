const puppeteer = require('puppeteer');
const devices = require('puppeteer/DeviceDescriptors');
const fs = require('fs');
const path = require('path');
const rimraf = require('rimraf');
const crypto = require("crypto");

describe('フォーム画像、フォームHTML、テキスト番号、テキストHTML、テキスト座標を集める', () => {
  it('1', async() => {
    let allUrls = ['http://test:3333/contact'];
    // for (const category of ["お問い合わせフォーム", "ユーザー登録", "会員登録", "登録フォーム"]) {
    //   const p = path.join(__dirname, `url/${category}.txt`);
    //   console.log(p);
    //   const urls = fs.readFileSync(p, { encoding: "utf-8" }).split("\n");
    //   allUrls.push(...urls);
    // }
    allUrls = allUrls.sort();
    const browser = await puppeteer.launch(
        // {headless: false}
    );
    for (const url of allUrls) {
      const md5hash = crypto.createHash('md5');
      md5hash.update(url);
      const hex = md5hash.digest("hex");
      const saveDir = path.join(__dirname, "../data/" + hex.slice(0, 8));
      if (fs.existsSync(saveDir)) {
        console.log("rm " + saveDir);
        rimraf.sync(saveDir);
      }
      await saveFormData(url, browser, saveDir);
    }
    await browser.close();
  }).timeout(3600000 * 24);
});

async function saveFormData(url, browser, saveDir) {
  console.log(url);
  const page = await browser.newPage();

  async function getRect(element, form) {
    const rect = await page.evaluate(element => {
      const { x, y, width, height } = element.getBoundingClientRect();
      return { x, y, width, height };
    }, element);
    if (form) {
      rect.x -= form.x;
      rect.y -= form.y;
    }
    return rect;
  }

  try {
    await page.emulate(devices['iPhone 8']);
    await page.goto(url, { waitUntil: "networkidle2" });
    const forms = await page.$$("form");

    for (let i = 0; i < forms.length; i++) {
      const form = forms[i];

      const formRect = await getRect(form);
      if (formRect.width === 0 || formRect.height === 0) {
        continue;
      }
      typeTexts = await form.$$("input[type=text]");
      if (typeTexts.length === 0) {
        continue;
      }

      if (!fs.existsSync(saveDir)) {
        fs.mkdirSync(saveDir);
      }

      const rects = [];
      for (let j = 0; j < typeTexts.length; j++) {
        const typeText = typeTexts[j];
        const rect = await getRect(typeText, formRect);
        rects.push(rect);
        await typeText.dispose();
      }
      const typeTextsPath = path.join(saveDir, `${i}.json`);
      console.log("save: " + typeTextsPath);
      fs.writeFileSync(typeTextsPath, JSON.stringify(rects, null, "  "));

      const imgPath = path.join(saveDir, `${i}.jpg`);
      console.log("save: " + imgPath);
      await page.screenshot({ path: imgPath, clip: formRect });

      await form.dispose();
    }
  } catch (e) {
    console.log(e);
  } finally {
    try {
      await page.close();
    } catch (e) {
      console.log(e);
    }
  }
}
