#!/usr/bin/env node
// Browser acceptance and screenshots — executed in GitHub Actions with Playwright.
// Uses only published source files; does not modify RTL, require a board or claim timing.
import {chromium} from "playwright";
import {resolve,dirname,join} from "node:path";
import {mkdir} from "node:fs/promises";
import {fileURLToPath,pathToFileURL} from "node:url";
import assert from "node:assert/strict";

const DIR=dirname(fileURLToPath(import.meta.url));
const PAGE=pathToFileURL(resolve(DIR,"index.html")).href;
const OUT=join(DIR,"visual-artifacts");
await mkdir(OUT,{recursive:true});
const browser=await chromium.launch({headless:true});
const errors=[];
async function openPage(viewport,url=PAGE){
 const page=await browser.newPage({viewport,deviceScaleFactor:1});
 page.on("pageerror",e=>errors.push(e.message));
 page.on("console",m=>{if(m.type()==="error")errors.push(m.text())});
 await page.goto(url,{waitUntil:"load"});
 await page.waitForFunction(()=>!!window.__openTPUAtlasChecks);
 return page;
}
const page=await openPage({width:1480,height:980});
const status=await page.evaluate(()=>({
 nodes:window.__openTPUAtlasChecks.nodes,
 flows:window.__openTPUAtlasChecks.flows,
 tour:window.__openTPUAtlasChecks.tour
}));
assert(status.nodes>=3700,"Hierarchy too shallow");
assert.equal(status.flows,12);assert.equal(status.tour,18);
await page.screenshot({path:join(OUT,"01-board-desktop.png"),fullPage:false});
assert(await page.locator("svg .part").count()>=2,"Board scene empty");
await page.locator('svg .part[aria-label="YPCB-00338 accelerator card"]').dblclick();
await page.waitForTimeout(150);
assert.equal(await page.evaluate(()=>window.__openTPUAtlasChecks.view()),"pciecard");
await page.screenshot({path:join(OUT,"02-fpga-card.png"),fullPage:false});
await page.evaluate(()=>window.__openTPUAtlasChecks.goto("slice"));
await page.screenshot({path:join(OUT,"03-slice.png"),fullPage:false});
await page.evaluate(()=>window.__openTPUAtlasChecks.goto("mxu"));
await page.screenshot({path:join(OUT,"04-mxu.png"),fullPage:false});
await page.evaluate(()=>window.__openTPUAtlasChecks.goto("array"));
await page.screenshot({path:join(OUT,"05-four-columns.png"),fullPage:false});
await page.evaluate(()=>window.__openTPUAtlasChecks.goto("col0g0pe0"));
await page.screenshot({path:join(OUT,"06-dsp48e1-pe.png"),fullPage:false});
await page.evaluate(()=>window.__openTPUAtlasChecks.playFlow("M"));
await page.locator("#btn-next").click();
await page.locator("#btn-next").click();
assert.equal(await page.evaluate(()=>window.__openTPUAtlasChecks.status().stage),2);
await page.screenshot({path:join(OUT,"07-dsp-flow-stage.png"),fullPage:false});
await page.locator("#btn-tour").click();
assert.equal(await page.evaluate(()=>window.__openTPUAtlasChecks.mode()),"tour");
await page.screenshot({path:join(OUT,"08-guided-tour.png"),fullPage:false});
const mobile=await openPage({width:390,height:844});
await mobile.evaluate(()=>window.__openTPUAtlasChecks.goto("tmem"));
await mobile.screenshot({path:join(OUT,"09-tmem-mobile.png"),fullPage:true});
assert(await mobile.locator("#node-title").isVisible(),"Mobile inspector invisible");
await mobile.close();
await page.close();await browser.close();
if(errors.length)throw Error("Browser errors:\n"+errors.join("\n"));
console.log("PASS Chromium visual acceptance: "+JSON.stringify(status));
console.log("PASS SVG double click, 4-column MXU, DSP stage player, tour and responsive mobile inspector.");
console.log("Screenshots stored under docs/silicon-explorer/visual-artifacts/");
