#!/usr/bin/env node
/*
 * Zero-dependency smoke test for the offline openTPU Deep Silicon Atlas.
 * Exercises the frontend with a minimal fake DOM; it is NOT a browser pixel/layout test,
 * a Verilator test or a Vivado timing validation.
 *
 * Run from repository root:
 *   node docs/silicon-explorer/smoke.mjs
 */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const text = fs.readFileSync(path.join(ROOT,"deep.html"),"utf8");
const script = text.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert(script, "Missing inline app");
const declared = [...text.split("<script>")[0].matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);
assert.equal(new Set(declared).size,declared.length,"Duplicate HTML ID");

class Element {
  constructor(id=""){
    this.id=id;this.children=[];this.attributes={};this.style={};this.dataset={};
    this.events={};this.textContent="";this.checked=id==="chash";
    this.value={
      a8:"-12",w8:"7",pcin:"100",act:"-23",acc:"100",trit:"0",
      tmaddr:"333",ddraddr:"8192",flowSelect:"0",pace:"950",
      metricSelect:"wall",params:"2",wbits:"4.25",bw:"17.056",util:"93",
      logicFn:"nand",logicA:"0",logicB:"0",logicCin:"0",
      lutMask:"AAAAAAAAAAAAAAAA",lutIndex:"19",
      ecap:"10",evolt:"1",efreq:"133.33",ealpha:".5"
    }[id]??"0";
    this.classList={add(){},remove(){},toggle(){}};
  }
  append(...v){this.children.push(...v)}
  replaceChildren(...v){this.children=[...v]}
  setAttribute(k,v){this.attributes[k]=v}
  addEventListener(k,fn){this.events[k]=fn}
  querySelectorAll(){return []}
  getBoundingClientRect(){return {width:1120,height:720}}
  setPointerCapture(){}
  releasePointerCapture(){}
  hasPointerCapture(){return false}
}
const elements=Object.fromEntries(declared.map(id=>[id,new Element(id)]));
const document={
  getElementById:id=>elements[id],
  createElement:tag=>new Element(tag),
  createElementNS:(_ns,tag)=>new Element(tag),
  querySelectorAll:()=>[]
};
const references=[...script.matchAll(/\$\("([^"]+)"\)/g)].map(x=>x[1]);
for(const id of references)assert(elements[id],"DOM reference is missing: "+id);
const history={replaceState(){}},location={hash:"#explore"},window={scrollTo(){}};
const app = new Function("document","window","location","history","addEventListener","setInterval","clearInterval",script);
app(document,window,location,history,()=>{},()=>1,()=>{});
assert.equal(elements.inspectorTitle.textContent,"openTPU Kintex-7");
assert(elements.scene.children.length>0);
assert.equal(elements.sourceGrid.children.length,25);
assert.match(elements.dspbits.textContent,/M = -84/);
assert.match(elements.ddrInfo.textContent,/physical channel 1/);
assert.equal(elements.logicOut.textContent,"1");
assert.equal(elements.truthRows.children.length,4);
assert.match(elements.lutBits.textContent,/Selected bit: 1/);

elements.logicFn.value="fa";
elements.logicA.value="1";
elements.logicB.value="1";
elements.logicCin.value="1";
elements.logicFn.events.change();
assert.equal(elements.logicOut.textContent,"1");
assert.equal(elements.logicCarry.textContent,"1");
assert.equal(elements.truthRows.children.length,8);

elements.search.value="c3g7p15dspmult";
elements.search.events.input();
assert.equal(elements.searchResults.children.length,1);
elements.searchResults.children[0].onclick();
assert.equal(elements.inspectorTitle.textContent,"Multiplier M");
elements.up.onclick();
assert.equal(elements.inspectorTitle.textContent,"DSP48E1 internal datapath");

elements.tmaddr.value="65535";
elements.tmaddr.events.input();
assert.match(elements.tmInfo.textContent,/bank 7; row 8191/);
elements.trit.value="3";
elements.trit.events.input();
assert.match(elements.ternaryResult.textContent,/INVALID/);
elements.lutMask.value="F0F0F0F0F0F0F0F0";
elements.lutIndex.value="2";
elements.lutMask.events.input();
assert.match(elements.lutBits.textContent,/Selected bit: 0/);

console.log("PASS openTPU Deep Silicon Atlas: DOM, initialization, 512-PE hierarchy, source links, DSP, ternary, memory, LUT6, CMOS and full-adder smoke tests.");
console.log("LIMIT: This fake DOM test does not verify SVG rendering, CSS, physical timing, browser accessibility or FPGA behavior.");
