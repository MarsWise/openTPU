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

const ST=(id,title,caption,src,kind="documented",tags=[])=>({id,title,caption,src:src||DATA.get(id)?.src,kind,tags});
const FLOWS=[
 {key:"1",name:"Boot / load",summary:"How a compiled openTPU program arrives in IMEM and starts the first instruction.",steps:[
 ST("host","Compile the program","The host compiles ISA commands and prepares the model image before any compute work is issued.","opentpu/compiler.py"),
 ST("hostpcie","Send over PCIe","XDMA transfers bytes and host commands into the board, which negotiates a concrete link width.",top),
 ST("shell","The FPGA shell","Host BAR0 control, LiteDRAM configuration and data paths are wired in the board wrapper.",top),
 ST("dma","Program loader","The slice's loader copies instruction chunks from external DRAM into IMEM while the compute core is held in reset.",sl),
 ST("imem","Instruction memory","A synchronous RAM stores ISA instructions. The first fetch has one additional initial pipeline cycle.",seq),
 ST("dispatch","Start issuing","The sequencer dispatches dependency-checked commands to independent engines.",seq)
 ]},
 {key:"2",name:"DDR stream",summary:"Why 64-byte beats, two channels and local memory matter more than adding arithmetic.",steps:[
 ST("drammap","Map a byte address","A 128-byte weight chunk is split into two 64-byte physical channel beats. CHASH can alter which half goes to which channel.",nat),
 ST("ddrbridge","Cross clock domains","Each DDR channel arbitrates accelerator and XDMA commands, keeping read-order tags and FIFO credits.",bridge),
 ST("bankctrl","Schedule memory requests","Port A scales, port B weight streams and scalar write traffic share the native adapter with backpressure.",nat),
 ST("weights","Prefetch into FIFOs","Scale and weight FIFOs let the MXU issuer run ahead of the consumer if dependency checks allow it.",mxu),
 ST("stationary","Reuse activations","Stationary ACT RAM rows are reused across weight advances; this is a separate structure from TMEM.",act),
 ST("epilogue","Write results","The post-MAC result passes scaling and conversion, then writes into local TMEM.",mxu)
 ]},
 {key:"3",name:"INT8 matmul",summary:"A matrix multiply travels from activation quantization through DSP48E1 chains to FP32 output.",steps:[
 ST("quant","Quantize activation rows","QACT prepares INT8 data and block scale metadata; no full-precision vector multiply is needed here.","rtl/vpu/otpu_quant.sv"),
 ST("actram","Keep activations stationary","The ACT RAM feeds up to four selected activation rows into the MXU per weight-chunk advance.",act),
 ST("weights","Fetch weight chunks","Each D=128 chunk of quantized weights moves from DDR3 through stream FIFOs.",mxu),
 ST("array","Four parallel columns","Four arithmetic columns each contain 128 logical PE positions, arranged as eight 16-position chains.",dsp,"derived",["512 MAC/cycle arithmetic ceiling"]),
 ST("col0g0","One sixteen-PE chain","The DSP48E1 multiplier output is combined with the preceding PE's PCIN to form a systolic dot-product partial sum.",dsp),
 ST("chend","Eight groups into one result","Each column end scales and adds the group sums through a nine-DSP cascade.",dsp),
 ST("fpconv","Scale and store","A 48-bit integer accumulator becomes FP32 with quantization scales before TMEM writeback.",mxu)
 ]},
 {key:"4",name:"BitNet / TMM",summary:"Follow two-bit ternary weights and see the difference between arithmetic peak and single-token decode.",steps:[
 ST("wpacked","Pack −1 / 0 / +1","Internal TMM codes are 00 for 0, 01 for +1, 10 for −1. Conversion from the original model representation matters.","docs/nextgen/BONSAI_4B.md"),
 ST("wdecode","Decode through integer logic","The packed weight selection feeds the DSP-based integer datapath; the precise FPGA mapping depends on TMM mode.",dsp),
 ST("col0g0pe0","One DSP product","A single logical PE uses DSP48E1 registers, multiplier and cascade output, not a magic ternary FLOP.",dsp),
 ST("groupScale","Optional GSC","Bonsai's grouped scales require extra logic and a matching firmware capability and bitstream.","docs/nextgen/BONSAI_4B.md"),
 ST("tmm","Decode bottleneck","Measured M=1 TMM reaches 127.7 MAC/cycle, while M=16 reaches 511.6 MAC/cycle. A single token does not fill the four-column datapath equally.","docs/nextgen/BITNET_COMPAT.md","measured"),
 ST("tmem","Finish in local memory","The scaled output is written to TMEM for the next layer's vector or matrix instruction.",mem)
 ]},
 {key:"5",name:"FP4 weights",summary:"The same integer matrix machinery handles low-bit weight chunks with format-specific scaling.",steps:[
 ST("dram","Load compact blocks","FP4/INT4 chunks halve the raw byte cost versus INT8, though scales and metadata still consume bandwidth.",nat,"derived"),
 ST("weights","Two blocks per 128-byte chunk","4-bit packing can carry two blocks in one chunk, with reuse controlled by format and PAIR flags.",mxu),
 ST("stationary","Read activation tiles","The MXU reuses INT8 activation blocks from ACT RAM.",act),
 ST("array","Compute in DSPs","Low-bit weight codes are decoded to products in the integer array with format-specific scale interpretation.",dsp),
 ST("fpconv","Numerical epilogue","Accumulated integer sums use FP32 scaling, not a direct native FP4 floating-point DSP operation.",mxu)
 ]},
 {key:"6",name:"Quantization",summary:"From FP32 TMEM to scaled INT8 stationary activation RAM.",steps:[
 ST("tmem","Input FP32 tile","The host or previous layer has placed activations into local FP32 tensor memory.",mem),
 ST("bankArb","Win bank access","The quantizer's TMEM reads compete with other units under the slice arbiter.",sl),
 ST("quant","Quantize with scale","QACT applies power-of-two offset/scaling and emits INT8 data plus block metadata.","rtl/vpu/otpu_quant.sv"),
 ST("scales","Write scale records","Each (row, block) stores a 64-bit record with the row scale and four offset fields.",act),
 ST("storage","Prepare stationary bytes","The ACT RAM stores quantized rows so the next MM can reuse them locally.",act)
 ]},
 {key:"7",name:"FP32 vector",summary:"A vector operation traverses TMEM banks, FP32 arithmetic lanes and a result stage.",steps:[
 ST("banks","Fetch contiguous words","The VPU requests lane-wide vectors from eight logical TMEM banks.",mem),
 ST("vpuLane","Eight SIMD elements","Simple arithmetic can be issued across eight lanes, with composite operations restricted to two long lanes.",vpu),
 ST("mul","Pipelined FP32 multiply","The multiplier rounds its FP32 result; its latency is at least two configured stages.",fpipe),
 ST("adder","Pipelined add","An add follows the multiplier for an FMADD-like instruction, with an additional rounding; it is not an IEEE fused FMA.",fpipe),
 ST("reduction","Optional vector reduction","RSUM, RSSQ and RDOT use interleaved partial sums and folding adders.", "rtl/vpu/otpu_vtree.sv"),
 ST("tmem","Store vector output","Result rows return to TMEM for the next instruction.",mem)
 ]},
 {key:"8",name:"Attention math",summary:"How sequential vector kernels and reductions can build an attention subgraph without adding an invented hardware unit.",steps:[
 ST("seq","Schedule the operator chain","The compiler expresses attention as ordinary MXU/VPU and memory instructions rather than a separate physical attention engine.",seq,"documented"),
 ST("mxu","Matrix products","Query/key/value projections and dot products use MXU instructions where the model/compiler supports them.",mxu),
 ST("reduction","Reduce scores","FP32 reduction hardware handles sums/maxima used in normalisation and related kernels.","rtl/vpu/otpu_vtree.sv"),
 ST("composite","Elementwise functions","EXP2, RECIP and other composite functions enable common softmax-style calculations through scheduled kernels.",vpu),
 ST("tmem","Intermediate data reuse","Intermediate matrices reside in TMEM when they fit; larger state can be streamed through DRAM.",mem)
 ]},
 {key:"9",name:"Bank conflict",summary:"Two engines asking for a shared TMEM bank cannot always advance in the same cycle.",steps:[
 ST("tmem","One logical scratchpad","TMEM is explicitly shared by DMA, VPU, MXU drain and quantizer. The bank count is eight.",mem),
 ST("bank3","Focus on bank 3","The logical bank index is the FP32 word address modulo eight. Every eighth word maps to this bank.",mem),
 ST("bankArb","Arbitrate a collision","The slice grants an engine only if all requested bank resources can be served, otherwise it holds; specific priority and replay are RTL-defined.",sl),
 ST("bypass","Preserve recent writes","A registered write bypass makes a read observe newly written data before physical BRAM storage catches up.",mem),
 ST("bram","Physical implementation","TMEM_IMPL=3 uses sub-banked RAM, read replay and 2× pumping. These details matter for timing and capacity.",mem)
 ]},
 {key:"0",name:"Resident decode",summary:"How instruction residency and on-card token handling can remove per-token host work.",steps:[
 ST("host","Prepare program once","Resident model execution compiles or loads reusable instruction programs and weight images beforehand.","docs/nextgen/BITNET_COMPAT.md"),
 ST("imem","Keep instruction buckets","Program code stays in IMEM until changed; a runtime can dispatch another decode position without reassembling instructions every time.",seq),
 ST("seq","Execute on the FPGA","The sequencer and unit commands run a full layer's compute operations while host communication is minimized.",seq),
 ST("vpu","Select the output token","The existing VPU can compute argmax over logits on the device, avoiding full logits transfer when supported.",vpu),
 ST("dispatch","HALT / CHAIN","The engine can chain to subsequent compiled programs while keeping model data resident.",seq),
 ST("hostpcie","Return token IDs","A resident generation loop need only send small results to the host, rather than all intermediate tensors.","docs/nextgen/BITNET_COMPAT.md","documented","Actual speedup must be confirmed with a matching resident-capable bitstream.")
 ]},
 {key:"B",name:"Host ↔ PCIe",summary:"The host/card path and why external bandwidth and protocol overhead influence end-to-end decoding.",steps:[
 ST("hostpcie","Host initiates transfer","Programs, model image and run arguments are prepared and sent over the host's PCIe interface.","opentpu/host/board.py"),
 ST("xdma","XDMA endpoint","The FPGA XDMA block receives data and exposes control registers.",top),
 ST("ddrbridge","Clock crossing","Asynchronous FIFOs allow accelerator and host streams to share each LiteDRAM channel.",bridge),
 ST("drammap","Memory chunk placement","Host DRAM data must respect the exact channel interleave and CHASH mapping.",nat),
 ST("shell","Observe completion","Counters, control and status registers report readiness and execution completion.","docs/observability.md")
 ]},
 {key:"M",name:"One DSP bit",summary:"The deepest source-backed arithmetic route: registers, DSP multiplier, adder and PCIN/PCOUT.",steps:[
 ST("col0","Column 0","One of four stationary activation rows processed per integer weight advance.",mxu),
 ST("col0g0","Sixteen-element chain","Products flow through a chain of sixteen Xilinx DSP48E1 technology primitives.",dsp),
 ST("col0g0pe0","PE 0","Each otpu_pe selects a weight stream, multiplies by the activation and accumulates.",dsp),
 ST("col0g0pe0areg","A/D inputs","A and D carry weight alternatives, selected by registered INMODE controls.",dsp),
 ST("col0g0pe0breg","BREG activation","The activation enters the DSP B register, avoiding external fabric multiply.",dsp),
 ST("col0g0pe0product","Multiply M","The DSP's product M is an actual Xilinx DSP48E1 datapath operation, not a generic LUT example.",dsp),
 ST("col0g0pe0psum","Accumulator P","P = PCIN + M (or P = M for FIRST); sums are kept on DSP cascade paths.",dsp),
 ST("col0g0pe0cascade","PCOUT","PCOUT carries the partial sum to the next PE. The exact cycle latency is controlled by instantiated DSP registers.",dsp),
 ST("logicgate","Under the primitive","Below DSP48E1, gates and individual MOSFETs can be shown as textbook circuit theory, not as a recovered Xilinx cell layout.",doc,"conceptual")
 ]}
];
const TOUR=[
 ["The whole machine","rack","The host, PCIe and FPGA form one accelerator system. Commands and weights must reach the card before computation begins."],
 ["The physical board","pciecard","The Inspur YPCB-00338 puts Kintex-7 compute beside two 2 GiB DDR3 channels."],
 ["The FPGA silicon","fpga","One Xilinx 7-series FPGA implements an explicit accelerator, not a RISC-V multicore CPU."],
 ["Memory channels","dram","The external memory path interleaves 64-byte beats across two DDR3 controllers."],
 ["The compute slice","slice","The openTPU slice contains seven principal interacting units and their memory interfaces."],
 ["Instruction scheduling","seq","A scoreboard allows independent instructions to overlap, while dependencies preserve correctness."],
 ["Data movement","dma","Moving data is a first-class hardware workload: the DMA shares DDR resources with the MXU."],
 ["Shared tensor SRAM","tmem","256 KiB of logical FP32 TMEM is divided into eight banks with arbitration and bypass."],
 ["Local activation tiles","actram","Quantized INT8 activations are stored stationary so matrix arithmetic can reuse them."],
 ["The matrix unit","mxu","D=128, MCOLS=4 — four physical dot-product columns and their accumulation machinery."],
 ["One column","col0","One stationary activation row spans eight chains of sixteen DSP48E1 PE positions."],
 ["One chain","col0g0","The dedicated DSP48E1 PCIN/PCOUT cascade reduces partial sums without a separate full-width fabric adder at every PE."],
 ["One processing element","col0g0pe0","A and D select weights; B carries the activation; M multiplies and P accumulates."],
 ["Ternary arithmetic","tmm","BitNet reuses the integer datapath; on-board performance depends on row count, not theoretical peak alone."],
 ["FP32 vector lanes","vpu","Eight FP32 lanes with two longer composite-function lanes complement the integer MXU."],
 ["Pipeline and reductions","fops","FP32 multiply and add are separate rounded operations. A reduction tree handles vector sums."],
 ["Physical logic","bitlogic","The RTL ends at mapped FPGA primitives. Going down to transistor cells becomes an explicitly labelled textbook explanation."],
 ["Evidence and limits","hostlens","Real instruction timing, stalls and traces belong to otpu-lens and board reports, not an artistic animation."]
].map(([title,id,caption])=>({title,id,caption}));
