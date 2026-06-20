// deck.html → presentation.pptx (슬라이드별 캡처 후 full-slide 이미지로 조립)
// 실행: node html2pptx.js [입력.html] [출력.pptx]
const puppeteer = require('puppeteer-core');
const PptxGenJS = require('pptxgenjs');
const path = require('path');
const fs = require('fs');

const IN = path.resolve(process.argv[2] || 'deck.html');
const OUT = path.resolve(process.argv[3] || 'presentation.pptx');
const CHROME = '/usr/bin/google-chrome-stable';
const W = 1280, H = 720;

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--force-color-profile=srgb'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 2 });
  await page.goto('file://' + IN, { waitUntil: 'networkidle0' });
  await page.evaluateHandle('document.fonts.ready');

  // 슬라이드를 정확히 1280x720 full-bleed로 강제(중앙 카드/그림자/다크배경 제거)
  await page.addStyleTag({ content: `
    html,body{background:#fff!important;margin:0!important}
    #stage{position:static!important;display:block!important;place-items:initial!important}
    .s{position:absolute!important;top:0!important;left:0!important;
       width:${W}px!important;height:${H}px!important;max-height:none!important;
       aspect-ratio:auto!important;box-shadow:none!important;transform:none!important}
    #prog,#pg,#tip,.ctrl{display:none!important}
    *{animation:none!important;transition:none!important}
  `});

  const n = await page.evaluate(() => document.querySelectorAll('.s').length);
  const pptx = new PptxGenJS();
  pptx.defineLayout({ name: 'W', width: 13.333, height: 7.5 });
  pptx.layout = 'W';

  const dir = path.join(path.dirname(OUT), '_shots');
  fs.mkdirSync(dir, { recursive: true });

  for (let i = 0; i < n; i++) {
    await page.evaluate((idx) => {
      document.querySelectorAll('.s').forEach((s, k) => {
        s.style.display = k === idx ? '' : 'none';
        s.classList.toggle('on', k === idx);
      });
    }, i);
    await new Promise(r => setTimeout(r, 350));
    const f = path.join(dir, `s${String(i + 1).padStart(2, '0')}.png`);
    await page.screenshot({ path: f, clip: { x: 0, y: 0, width: W, height: H } });
    const slide = pptx.addSlide();
    slide.addImage({ path: f, x: 0, y: 0, w: 13.333, h: 7.5 });
    process.stdout.write(`  slide ${i + 1}/${n}\n`);
  }
  await browser.close();
  await pptx.writeFile({ fileName: OUT });
  console.log('[saved] ' + OUT);
})().catch(e => { console.error(e); process.exit(1); });
