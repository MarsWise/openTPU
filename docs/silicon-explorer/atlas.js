"use strict";
/*
 * openTPU Atlas v2 — independently authored visualization, no external libraries.
 * Numbers are based on openTPU's pinned target and linked documentation.
 * Animated flows are explanations, NOT VCD/SAIF traces or timing measurements.
 */
(()=> {
const REV="55b975c14346cc692f1765729d750d79e8bc0b04";
const SOURCE="https://github.com/MarsWise/openTPU/blob/"+REV+"/";
const NS="http://www.w3.org/2000/svg";
const $=id=>document.getElementById(id);
const E=(tag,props={},parent)=>{
 const el=document.createElementNS(NS,tag);
 for(const [k,v] of Object.entries(props))el.setAttribute(k,String(v));
 if(parent)parent.appendChild(el);return el;
};
const text=(parent,x,y,value,size=15,fill="#f1f5f9",weight=400)=>{
 let t=E("text",{x,y,"font-size":size,fill,"font-weight":weight},parent);t.textContent=String(value);return t;
};
const html=(tag,cls="",content)=>{
 const n=document.createElement(tag);if(cls)n.className=cls;if(content!==undefined)n.textContent=String(content);return n;
};
const enc=encodeURIComponent;
const DATA=new Map();
function node(id,parent,name,desc,short,type,src,facts={},status="documented",note="",extra=[]){
 if(DATA.has(id))throw Error("duplicate "+id);
 if(parent&&!DATA.has(parent))throw Error("unknown parent "+id+" -> "+parent);
 const n={id,parent,name,desc,short:short||name,type:type||"control",src,extra,
  facts,status,note,children:[]};
 DATA.set(id,n);
 if(parent)DATA.get(parent).children.push(id);
 return n;
}
const doc="docs/targets.md",brd="targets/ypcb-00338.toml",top="rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv";
const sl="rtl/top/otpu_slice.sv",seq="rtl/seq/otpu_seq.sv",dma="rtl/dma/otpu_dma.sv";
const mxu="rtl/mxu/otpu_mxu.sv",dsp="rtl/tech/xilinx7/otpu_tech_dsp.sv";
const mem="rtl/mem/otpu_tmem.sv",act="rtl/mem/otpu_actram.sv",vpu="rtl/vpu/otpu_vpu.sv";
const fpipe="rtl/vpu/otpu_fpipe.sv",ram="rtl/tech/generic/otpu_tech_ram.sv";
const nat="rtl/mem/otpu_native_dram.sv",bridge="rtl/shell/otpu_mem_ch.sv";
node("rack",null,"AI workstation and PCIe card","The host compiles the model into openTPU programs, manages weights in DDR3, transfers commands and reads responses. The target here is the Inspur YPCB-00338 card.","Host + FPGA card","control",doc,{"Scope":"one FPGA card","Model data":"host ↔ DDR3"},"documented","System grouping and positions are logical, not a photo or a rack footprint.");
node("host","rack","Host machine","Host-side Python engine and device driver configure the board, assemble programs and coordinate PCIe transfers.","CPU + RAM + runtime","control","opentpu/host/board.py",{"Software":"Python runtime","Role":"orchestration"},"documented");
node("pciecard","rack","YPCB-00338 accelerator card","Xilinx Kintex-7 FPGA, two 2 GiB DDR3 channels, PCIe Gen2 x8 target, and clock/power components.","FPGA + dual DDR3","compute",brd,{"FPGA":"XC7K480T","DRAM":"4 GiB DDR3","Host":"PCIe Gen2 x8","Clock":"133.33 MHz"},"documented","Board bitstreams may negotiate a lower PCIe width; target configuration is not proof of link negotiation.");
node("hostpcie","host","Host PCIe controller","Transfers model chunks and control writes to the FPGA. The XDMA endpoint can be constrained by the negotiated link.","Host I/O","control","docs/board.md");
node("runtime","host","Compiler and runtime","Program ISA compilation, tensor layouts, conversion of model weights, inference and host-device synchronization.","Python kernels","control","opentpu/compiler.py",{"Execution":"compiled instruction programs"},"documented");
node("hostlens","host","Trace inspection","The otpu-lens profiler is the actual trace timeline and stall analysis tool, complementary to this explanatory atlas.","otpu-lens","control","docs/lens.md",{"Input":"profile / trace"},"documented","This atlas does not claim to import real hardware traces.");
node("fpga","pciecard","Kintex-7 FPGA die","The active RTL implements one compute slice with a systolic MXU, SIMD VPU, quantizer, DMA, TMEM and sequencer.","XC7K480T","compute",top,{"Family":"Xilinx 7-series","Process":"28 nm family","Target":"-2 speed grade"},"documented","The physical layout inside the die is not derivable from the source's block hierarchy.");
node("ddr0","pciecard","DDR3 channel 0","One 64-bit 2 GiB DDR3 channel behind LiteDRAM.","2 GiB · 64b","memory",bridge,{"Capacity":"2 GiB","Rate":"1066 MT/s"},"documented");
node("ddr1","pciecard","DDR3 channel 1","Second independent 64-bit 2 GiB DDR3 channel.","2 GiB · 64b","memory",bridge,{"Capacity":"2 GiB","Rate":"1066 MT/s"},"documented");
node("powers","pciecard","Power and clocks","Board-level power regulation, 50 MHz reference input and MMCM-generated clocks.","Clock / regulator","physical",top,{"Input":"50 MHz","Core":"133.33 MHz"},"conceptual","The power regulator topology is not reconstructed from board schematics.");
node("shell","fpga","FPGA shell","XDMA/AXI, board control, LiteDRAM, clock domain crossings, status counters, trace buffers and native DRAM arbitration.","Host-facing shell","control","rtl/shell/otpu_board.sv");
node("slice","fpga","openTPU compute slice","Independent hardware engines run concurrently under dependency-aware command issue, sharing TMEM and DRAM channels.","One slice","compute",sl,{"Slices":"S=1","Core":"133.33 MHz","DMAs":"explicit","Units":"SEQ / DMA / MXU / VPU / QUANT"},"documented");
node("xdma","shell","XDMA / PCIe endpoint","The PCIe host sends configuration and data over Xilinx XDMA and AXI interfaces.","PCIe endpoint","control",top,{"Target link":"Gen2 x8"},"documented");
node("clocks","shell","Clock / reset domains","Different FPGA, DDR3 and PCIe domains connected by asynchronous FIFO structures; TMEM can use doubled core clock.","MMCM + CDC","control",top,{"Core":"133.33 MHz","TMEM pump":"2× core"},"documented");
node("control","shell","Registers / trace","Control, IDs, trace and observability registers for host-visible status.","BAR0 + trace","control","docs/observability.md",{"Identity":"CFG_HASH / BUILD_ID"},"documented");
node("dram","shell","Native memory adapter","Interleaves 64-byte beats over two channels, CHASH swaps chunk halves for better stride distribution.","64 B beats","memory",nat,{"Channels":"2","D":"128 bytes","Hash":"CHASH"},"documented");
node("ddrbridge","dram","DDR3 bridges","Each channel merges accelerator native traffic and XDMA AXI host traffic, with FIFOs for clock crossings.","otpu_mem_ch","memory",bridge);
node("drammap","dram","Channel interleave","A 128-byte chunk spans two physical channels: consecutive 64-byte beats map to channel parity, optionally XOR'ed with chunk index parity.","CHASH map","memory",nat,{"Beat":"64 bytes","Chunk":"128 bytes"},"documented");
node("bankctrl","dram","DRAM arbiter","Port A handles scales and scalar traffic; port B streams chunks. Reads and writes are queued with backpressure and tags.","Ports A / B / SW","memory",nat,{"Order":"per-channel queues"},"documented");
node("seq","slice","Instruction sequencer","R0–R15 register setup, IMEM fetch, dependency scoreboard and dispatch of independent commands into parallel execution engines.","ISA + scoreboard","control",seq,{"Issue":"in order","Window":"16 entries (target)"},"documented");
node("dma","slice","DMA engine","Explicit movement between external DDR3 and TMEM; priority over the MXU for one DRAM traffic path.","LD / ST / DSTEP","control",dma);
node("mxu","slice","Systolic MXU","Integer matrix multiplication with stationary activations and streamed quantized weights, exact accumulation and FP32 epilogue.","D128 × 4 columns","compute",mxu,{"D":"128","Columns":"4","Peak":"512 integer MAC/cycle"},"derived","512 MAC/cycle is an arithmetic ceiling, not the sustained throughput of every model.");
node("vpu","slice","Vector processing unit","Eight FP32 lanes, including two lanes equipped to execute composite transcendental functions.","8 FP32 lanes","compute",vpu,{"LANES":"8","VPU_CL":"2"},"documented");
node("quant","slice","Activation quantizer","QACT creates INT8 blocks and scale metadata from TMEM FP32 activations for the stationary ACT RAM.","QACT","compute","rtl/vpu/otpu_quant.sv");
node("tmem","slice","Tensor memory","FP32 scratchpad implemented with eight logical banks, arbitration, bypass and resource-specific FPGA memory structures.","256 KiB · 8 banks","memory",mem,{"Words":"65,536","Word":"32 bits","Banks":"8"},"documented");
node("actram","slice","Activation RAM","Stationary INT8 tiles and per-block scale records feeding the MXU across its four columns.","16 × 128 × 128","memory",act,{"Raw bytes":"256 KiB","Rows":"16","Blocks":"128"},"derived","Raw INT8 capacity excludes scale metadata and RAM mapping overhead.");
node("collective","slice","Collective unit","Cross-slice BAR/GATHER RTL exists; the current YPCB target has one slice, not a physical multi-slice mesh.","BAR / GATHER","control","rtl/top/otpu_coll.sv",{"S":"1 (on card)"},"documented");
node("imem","seq","Instruction memory","Synchronous FPGA RAM stores compiled instruction words. An instruction is fetched one ahead into a register.","IMEM","memory",seq,{"IMEM words":"32,768"},"documented");
node("reg","seq","Register file","Scalar instruction arguments R8..R15 and register-relative addresses, loop counts and HALT/CHAIN logic.","R0…R15","control",seq);
node("scoreboard","seq","Dependency scoreboard","Tracks memory footprints and conflicting RAW/WAR/WAW accesses so independent DMA, MXU and VPU commands can overlap.","RAW / WAR / WAW","control",seq,{"Policy":"oldest ready per unit"},"documented");
node("dispatch","seq","Issue and release","Starts independent units; matrix prefetch can start earlier than the release of stationary operands.","Command window","control",seq);
node("preload","dma","LD / ST engine","Loads chunk streams from DDR3 into TMEM and stores results with explicit completion.","Transfers","control",dma);
node("stream","dma","Stream engine","DSTEP/STREAM use the DMA's data stream and the VPU arithmetic tail.","DSTEP / STREAM","control",dma);
node("waitw","dma","WAITW rendezvous","Memory-polling synchronization command, with timeout/error behavior.","WAITW","control",dma);
node("weights","mxu","Weight and scale streamer","B port streams quantized weight chunks; A port fetches scale rows; FIFOs decouple fetch from compute.","DDR3 → FIFO","memory",mxu,{"Chunk":"128 bytes"},"documented");
node("stationary","mxu","Stationary ACT tiles","One weight chunk can be consumed against multiple activation rows held in ACT RAM, depending on M and column reuse.","Reuse ACT RAM","memory",act);
node("array","mxu","Four systolic columns","Each of four columns performs a chain of 128 logical products, divided into 8 groups of 16 DSP48E1 processing elements.","4 × 128 PE","compute",dsp,{"Array products":"512 / cycle (peak)","PE cell":"DSP48E1"},"derived","Pipelined product count includes logical arithmetic positions; complete FPGA DSP count additionally includes column end, VPU and infrastructure.");
node("epilogue","mxu","Column-end and FP32","Eight group sums are scaled and accumulated in a nine-DSP column end, then converted to FP32 with quantization scales.","Exact sum → FP32","compute",dsp,{"DSPs / end":"9","Exact accumulator":"48 bits"},"documented");
node("tmm","mxu","Ternary / BitNet matrix","Ternary packing carries values −1/0/+1 using 2-bit codes. Branch-specific GSC handles group scales; effective decode throughput depends on M.","TMM / GSC","compute",mxu,{"TMM M=1":"127.7 MAC/cycle","TMM M=16":"511.6 MAC/cycle"},"measured","These are dated FPGA measurements from the BitNet compatibility report.",["docs/nextgen/BITNET_COMPAT.md","docs/nextgen/BONSAI_4B.md"]);
node("wpacked","tmm","Packed ternary weights","Internal TMM uses 00=0, 01=+1, 10=−1. Bits from source checkpoints may need lossless format conversion.","2-bit codes","memory","docs/nextgen/BONSAI_4B.md",{"Valid":"00,01,10","Reserved":"11"},"documented");
node("wdecode","tmm","Ternary decode / select","Selectable encoded weight groups feed an integer MAC datapath and produce scaled results.","No FP32 weight multiply","compute",dsp,{},"documented","The DSP RTL implements packed weight selection; the whole end-to-end TMM path is version-dependent.");
node("groupScale","tmm","Group scale adapter","Bonsai GSC accounts for per-block scale before FP32 accumulation; documented verification is not identical to production board.","GSC path","compute","docs/nextgen/BONSAI_4B.md",{},"documented","Do not mark new GSC as board-verified without a matching bitstream.");
node("vpuLane","vpu","Eight SIMD lanes","LANES=8 vector FP32 elements per cycle; composite functions are scheduled onto the configured long lanes.","8 lanes","compute",vpu,{"Ordinary op":"8 elements/cycle (peak)","Composite lanes":"2"},"derived");
node("fops","vpu","FP32 pipelined arithmetic","FP32 add/mul are independently pipelined; openTPU fmadd is mul then add with two roundings, not a single fused FMA.","mul → add","compute","rtl/vpu/otpu_fpipe.sv",{"Mul latency":"≥2 cycles","Add latency":"≥4 cycles"},"documented");
node("reduction","vpu","Reduction network","RSUM/RDOT use 64 interleaved partials followed by scheduled folding stages.","64 partials","compute","rtl/vpu/otpu_vtree.sv");
node("composite","vpu","Composite function pipeline","EXP2, RECIP, RSQRT and LOG2 use staged numerical approximations and reusable slots.","Horner / Newton","compute","rtl/vpu/otpu_se_comp.sv");
node("rounding","fops","FP32 rules","Binary32 rounding-to-nearest-even and flush-to-zero on input/output. Source defines exact behavior.","FP32 FTZ","control","rtl/vpu/otpu_fp.sv",{"Round":"nearest-even","Subnormals":"flush to zero"},"documented");
node("mul","fops","FP32 multiplier","Staged arithmetic computes a rounded FP32 product.","FP32 multiply","compute","rtl/vpu/otpu_fpipe.sv");
node("adder","fops","FP32 adder","Staged FP32 addition with normalization and rounding behavior specified in the source.","FP32 add","compute","rtl/vpu/otpu_fpipe.sv");
node("tree","reduction","Partial folding","The 64 partials are folded by pipelined adders and delayed registers. Exact scheduling follows RTL.","Pipelined fold","compute","rtl/vpu/otpu_vtree.sv");
node("e2","composite","EXP2 polynomial","Polynomial Horner steps and a range-reduction path approximate 2^x.","9 operations","compute","rtl/vpu/otpu_se_comp.sv",{"Ops":"9"},"documented");
node("rrecip","composite","RECIP iterative","Newton refinement reuses the FP32 multiplier and adder path.","6 operations","compute","rtl/vpu/otpu_se_comp.sv",{"Ops":"6"},"documented");
node("rsqrt","composite","RSQRT iterative","Reciprocal square-root approximation uses a longer series of arithmetic operations.","10 operations","compute","rtl/vpu/otpu_se_comp.sv",{"Ops":"10"},"documented");
node("log","composite","LOG2 polynomial","Polynomial and exponent extraction implement logarithm approximation.","10 operations","compute","rtl/vpu/otpu_se_comp.sv",{"Ops":"10"},"documented");
node("rom","composite","TFN tables","An optional table-based path stores coefficient triples in ROM; target capabilities determine its availability.","Coefficient ROM","memory","rtl/vpu/otpu_tfn_rom.sv",{"Entries":"2,048"},"documented");
node("bankArb","tmem","TMEM request arbiter","Contending units are granted whole-unit access to banked memory. Requests not granted can stall or replay.","Priority + replay","control",sl);
node("banks","tmem","Eight TMEM banks","bank = (FP32 word address) mod 8, 8,192 words per logical bank.","Bank decoder","memory",mem,{"Banks":"8","Words/bank":"8192"},"documented");
node("bram","tmem","Block RAM implementation","TMEM_IMPL=3 uses sub-bank organization and a 2× pumped storage path with forwarding on registered writes.","Xilinx BRAM","memory",mem,{"Mode":"TMEM_IMPL=3","Pump":"2×"},"documented");
node("bypass","tmem","Write bypass","Registered write pipeline with bypass keeps same-address read-after-write behavior observable in the core.","Forwarding","control",mem);
node("storage","actram","Activation byte arrays","Each column/piece has physical block RAM storage for stationary INT8 activation blocks.","INT8 arrays","memory",act);
node("scales","actram","Scale records","Each (row, block) stores Sa and four sub-block offsets as a 64-bit record.","Sa + offsets","memory",act,{"Record":"64 bits"},"documented");
node("chend","epilogue","Column-end nine DSPs","Each column end consumes eight group sums and their block scales, using cascaded DSP48E1 cells.","9 DSP48E1","compute",dsp);
node("fpconv","epilogue","FP32 output conversion","Exact int48 output converted and scaled, optionally accumulated, before TMEM writeback.","i2f + scaling","compute",mxu);
for(let b=0;b<8;b++){
 node("bank"+b,"banks","TMEM bank "+b,"Logical FP32 bank "+b+" accepts word addresses with modulo-eight result "+b+".","8,192 words","memory",mem,{"Selector":"addr mod 8 = "+b,"Storage":"8,192 words"},"documented");
 node("sub"+b,"bank"+b,"Sub-banks and ports","The 2× pumped implementation subdivides physical RAM storage and arbitrates resource access; memory sources define port behavior.","BRAM ports","memory",mem,{"Physical exact placement":"not provided"},"documented");
 node("cell"+b,"sub"+b,"SRAM cell (textbook)","A conventional static SRAM bit cell may be represented as cross-coupled inverters and two access transistors.","6T cell","physical",ram,{"Cell model":"6T SRAM"},"conceptual","A textbook six-transistor SRAM drawing, NOT a disclosed Xilinx BRAM cell schematic.");
}
for(let c=0;c<4;c++){
 const col="col"+c;
 node(col,"array","Column "+c,"A stationary INT8 activation row takes one weight chunk across 128 logical processing positions.","128 MAC positions","compute",dsp,{"D":"128","Groups":"8","Chain positions":"16"},"documented");
 for(let g=0;g<8;g++){
  const grp=col+"g"+g;
  node(grp,col,"DSP chain "+g,"Sixteen DSP48E1 positions share a PCIN/PCOUT cascade across one group.","16 PE","compute",dsp,{"Group":String(g),"Products":"16"},"documented");
  for(let p=0;p<16;p++){
   const pe=grp+"pe"+p;
   node(pe,grp,"PE "+p,"Processing element otpu_pe: signed input registers, DSP48E1 hardware product and cascade accumulation.","DSP48E1","compute",dsp,{"Column":String(c),"Group":String(g),"Index":String(p)},"documented");
   for(const [sid,name,brief] of [
    ["areg","A / D weight inputs","Two register paths carry selectable packed weight components."],
    ["inmode","INMODE select","A registered control selects the input presented to the multiplier."],
    ["breg","BREG activation","The signed activation is held in a DSP input register."],
    ["product","M multiplier","Signed product created inside the DSP48E1 dedicated arithmetic primitive."],
    ["psum","P accumulator","P = M at the first position, otherwise P = PCIN + M."],
    ["cascade","PCOUT cascade","Dedicated cascade output feeds the following processing element."]
   ]) node(pe+sid,pe,name,brief,"DSP primitive","compute",dsp,{"Primitive":"DSP48E1","Field":sid},"documented","A technology-level functional description, not an extracted transistor netlist.");
  }
 }
}
node("bitlogic","fpga","Logic and FPGA primitives","An educational expansion below synthesis: DSP, LUT6, CARRY4, BRAM and flop examples. Exact instantiated count requires an implementation report.","Technology cells","physical","docs/board.md",{},"conceptual","The blocks below are generic primitive descriptions, not a generated LUT-level netlist.");
node("lut","bitlogic","LUT6 logic","A six-input LUT evaluates one of 64 configuration bits as a Boolean function.","64-bit truth table","physical",doc,{"Inputs":"6","Truth entries":"64"},"conceptual");
node("carry","bitlogic","CARRY4 chain","Dedicated fast FPGA carry propagation for additions and counters.","Carry chain","physical",doc,{"Primitive":"Xilinx 7-series"},"conceptual");
node("flop","bitlogic","Flip-flop / CE","Clocked bit storage with explicit clock enable. Different reset/enable forms map to suitable primitives.","FF + enable","physical",doc,{},"conceptual");
node("logicgate","bitlogic","Boolean gates","XOR/AND/NAND, mux and full-adder Boolean equations explain digital functions below RTL.","Logic gates","physical",doc,{},"conceptual");
node("sramcell","bitlogic","SRAM 6T cell","Cross-coupled inverter storage cell, with access transistors, as a generic technology illustration.","6 transistors","physical",doc,{"Transistors":"6 (textbook)"},"conceptual","Xilinx does not disclose this exact SRAM cell from openTPU source.");
node("fet","logicgate","CMOS MOSFET","An idealized transistor switches conduction through its gate. This is not a transistor extracted from Kintex-7.","Textbook MOSFET","physical",doc,{"Approximate scale":"tens of nm"},"conceptual");
node("silicon","fet","Silicon lattice","A semiconductor crystal and dopant control carrier behavior. Illustrated symbolically.","Si crystal","physical",doc,{},"conceptual");
node("atom","silicon","Silicon atom","A silicon atom has 14 protons and 14 electrons when neutral. The illustration is deliberately schematic.","Atomic structure","physical",doc,{"Atomic number":"14"},"conceptual");
node("nucleus","atom","Atomic nucleus","Protons and neutrons bound inside a nucleus. No openTPU implementation detail exists at this scale.","Nuclear scale","physical",doc,{},"conceptual");
node("quarks","nucleus","Quarks in nucleons","A proton's valence quark content is uud; this is a physics education endpoint, not a chip design.","u u d","physical",doc,{"Proton":"2 up + 1 down"},"conceptual","Deep physics does not provide direct circuit-level performance evidence.");
node("ddrchip","ddr0","DDR3 memory device","Off-chip DRAM storage and interface; physical vendor package layout is not reconstructed.","DDR3 package","physical","docs/litedram.md",{},"conceptual");
node("dramBank","ddrchip","DRAM bank","DRAM controller groups rows and columns across banks. No exact vendor row address floorplan is inferred.","DRAM bank","memory","docs/litedram.md");
node("dramRow","dramBank","DRAM row","A row of capacitor cells accessed through wordlines and sense amplifiers. Explanatory representation.","Wordline + sense","physical","docs/litedram.md",{},"conceptual");
node("dramCell","dramRow","1T1C DRAM cell","A textbook storage capacitor charged/discharged through one access transistor.","One transistor + capacitor","physical","docs/litedram.md",{"Cell":"1T1C"},"conceptual","Not a photomicrograph or cell extraction of this DDR3 part.");
const PATHNODES=()=>DATA.size;
