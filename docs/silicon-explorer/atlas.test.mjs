#!/usr/bin/env node
// No-dependency smoke tests: DOM initialization, all 12 flows, 18-tour,
 // every source-backed zoom level, and the native profile loader.
 // This does not substitute for a real-browser screenshot or FPGA measurements.
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import assert from "node:assert/strict";
const DIR=path.dirname(fileURLToPath(import.meta.url));
const html=fs.readFileSync(path.join(DIR,"index.html"),"utf8");
const js=fs.readFileSync(path.join(DIR,"atlas.js"),"utf8");
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);
assert.equal(new Set(ids).size,ids.length,"Duplicate HTML IDs");
for(const name of ["flow-nav","diagram","inspector","player","profile-file","pf-event","btn-tour","node-search"])
 assert(ids.includes(name),"Missing UI element "+name);
new Function(js); // Parse check.
const events={};
class El {
 constructor(id=""){this.id=id;this.tagName=id.toUpperCase();this.children=[];this.attributes={};
  this.style={};this.dataset={};this.events={};this.textContent="";this.value="";
  this.hidden=false;this.disabled=false;this.classList={add(){},remove(){},toggle(){}};}
 appendChild(n){this.children.push(n);return n}
 append(...ns){this.children.push(...ns)}
 replaceChildren(...ns){this.children=[...ns]}
 setAttribute(k,v){this.attributes[k]=v}
 addEventListener(k,fn){this.events[k]=fn}
 getBoundingClientRect(){return {width:1120,height:720}}
 scrollIntoView(){}
 querySelectorAll(){return []}
 click(){this.onclick?.()}
 setPointerCapture(){}
 releasePointerCapture(){}
}
const DOM=Object.fromEntries(ids.map(id=>[id,new El(id)]));
const document={
 body:new El("body"),documentElement:new El("html"),
 getElementById:id=>DOM[id],createElement:tag=>new El(tag),
 createElementNS:(_ns,tag)=>new El(tag),querySelectorAll:()=>[],
 addEventListener:(name,fn)=>events["document."+name]=fn};
let hash="";
const location={get hash(){return hash},set hash(v){hash=v},href:"https://example.test/openTPU/"};
const history={replaceState:(_a,_b,h)=>{hash=h}};
const window={scrollTo(){},addEventListener:(name,fn)=>events["window."+name]=fn};
new Function("document","window","location","history","navigator","setInterval","clearInterval","setTimeout","clearTimeout",js)(
 document,window,location,history,{clipboard:{writeText:async()=>{}}},()=>1,()=>{},()=>1,()=>{});
const api=window.__openTPUAtlasChecks;
assert(api,"Test interface missing");
assert.equal(api.flows,12);
assert.equal(api.tour,18);
assert(api.nodes>=3700,"Unexpectedly shallow hierarchy");
assert.equal(DOM["flow-nav"].children.length,12);
assert.equal(api.view(),"rack");
const locations=[
 "pciecard","fpga","shell","slice","seq","dma","mxu","array","col0",
 "col0g0","col0g0pe0","col0g0pe0product","col3g7pe15cascade",
 "tmem","banks","bank7","cell7","ddr0","ddrchip","dramRow","dramCell",
 "vpu","vpuLane","vlane0","fops","reduction","quant","scoreboard",
 "bitlogic","logicgate","fet","silicon","atom","nucleus","quarks"
];
for(const id of locations){assert(api.getNode(id),"Missing node "+id);api.goto(id);
 assert.equal(api.view(),id);assert(DOM.camera.children.length>0,"SVG scene missing for "+id);}
for(const key of ["1","2","3","4","5","6","7","8","9","0","B","M"]){api.playFlow(key);
 assert(api.status().steps>=5,"Flow too shallow "+key);assert(DOM["stage-buttons"].children.length>=5);
 DOM["btn-next"].onclick();assert.equal(api.status().stage,1);}
DOM["btn-tour"].onclick();assert.equal(api.mode(),"tour");
assert.equal(DOM["stage-buttons"].children.length,18);
DOM["btn-next"].onclick();assert.equal(api.view(),"pciecard");
DOM["node-search"].value="col3g7pe15";
DOM["node-search"].events.input();
assert(DOM["search-results"].children.length>=1);
DOM["search-results"].children[0].onclick();
assert.equal(api.view(),"col3g7pe15");
DOM["btn-up"].onclick();assert.equal(api.view(),"col3g7");
const profile={format:"openTPU-profile",version:1,profiles:[{
 kind:"rtl",name:"Synthetic fixture (not a board measurement)",cycles:200,clock_mhz:133.33,
 units:["DMA","MXU","QUANT","VPU","COLL"],
 instrs:[
 [0,0,12,0,"LD","8 words",0,0,-1,12,24,0,{},12,1,0],
 [0,1,15,1,"MM","1 row × 128 block",3,3,21,25,61,0,{},36,1,1],
 [0,2,18,3,"VOP","ADD",6,6,-1,65,83,0,{},18,0,0]
 ],sources:["fixture:unit_test"],roofline:{bound:30,efficiency:.15}}]};
api.testProfile(profile);
assert.equal(api.profileState().count,3);
assert.equal(DOM["pf-name"].textContent,"Synthetic fixture (not a board measurement)");
assert.equal(api.view(),"dma");
DOM["pf-next"].onclick();assert.equal(api.view(),"mxu");
assert(DOM["pf-event"].children.length>=3,"Missing measured event metadata");
console.log("PASS openTPU immersive atlas:",api.nodes,"nodes,",api.flows,"flows,",api.tour,"tour steps,",locations.length,"deep scenes, source-grounded profile parser.");
console.log("LIMIT: Fake-DOM tests do not prove layout, route simulation, power, Vivado timing, or physical FPGA output.");
