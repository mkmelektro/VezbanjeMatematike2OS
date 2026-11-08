import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/ort.min.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/";

const NUMBER_ALLOWED = new Set([
  0,1,2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,22,50,53,69,78,82,83,110,112
]);
const REPEAT_LIMIT = 3;
let enc = null, dec = null;

async function init(msg){
  try{
    const [eb,db]=await Promise.all([
      fetch(msg.encoderUrl).then(r=>{if(!r.ok)throw Error("encoder HTTP "+r.status);return r.arrayBuffer()}),
      fetch(msg.decoderUrl).then(r=>{if(!r.ok)throw Error("decoder HTTP "+r.status);return r.arrayBuffer()})
    ]);
    const opts={executionProviders:["wasm"]};
    [enc,dec]=await Promise.all([
      ort.InferenceSession.create(eb,opts),
      ort.InferenceSession.create(db,opts)
    ]);
    self.postMessage({type:"init-done"});
  }catch(e){self.postMessage({type:"init-error",error:String(e&&e.stack||e)})}
}
function logSoftmax(logits,offset,size){
  const out=new Float64Array(size);let max=-Infinity;
  for(let i=0;i<size;i++){const v=logits[offset+i];if(v>max)max=v}
  let sum=0;for(let i=0;i<size;i++)sum+=Math.exp(logits[offset+i]-max);
  const ls=Math.log(sum);for(let i=0;i<size;i++)out[i]=logits[offset+i]-max-ls;
  return out;
}
function mask(logp,size,mode){
  if(mode!=="number")return;
  for(let i=0;i<size;i++)if(!NUMBER_ALLOWED.has(i))logp[i]=-Infinity;
}
async function recognize(m){
  try{
    if(!enc||!dec)throw Error("worker model not initialized");
    const t0=performance.now();
    const pixels=new ort.Tensor("float32",m.tensor,[1,1,m.height,m.width]);
    const pmask=new ort.Tensor("bool",m.mask,[1,m.maskHeight,m.maskWidth]);
    const er=await enc.run({pixel_values:pixels,pixel_mask:pmask});
    pixels.dispose();pmask.dispose();
    const ef=er.encoder_features, em=er.encoder_mask;
    const t1=performance.now();

    const ids=[m.sos];let last=-1,repeat=0;
    for(let step=0;step<m.maxSteps;step++){
      const input=new ort.Tensor("int64",BigInt64Array.from(ids.map(BigInt)),[1,ids.length]);
      const rr=await dec.run({encoder_features:ef,encoder_mask:em,input_ids:input});
      input.dispose();
      const logits=rr.logits.data;
      const off=(ids.length-1)*m.vocabSize;
      const lp=logSoftmax(logits,off,m.vocabSize);mask(lp,m.vocabSize,m.mode);
      let best=0,bv=-Infinity;
      for(let i=0;i<m.vocabSize;i++){if(lp[i]>bv){bv=lp[i];best=i}}
      if(best===m.eos)break;
      if(best===last){repeat++;if(repeat>=REPEAT_LIMIT)break}else repeat=0;
      last=best;ids.push(best);
    }
    ef.dispose();em.dispose();
    const t2=performance.now();
    self.postMessage({
      type:"recognize-done",id:m.id,ids:ids.slice(1),
      encoderMs:Math.round(t1-t0),decoderMs:Math.round(t2-t1),totalMs:Math.round(t2-t0)
    });
  }catch(e){self.postMessage({type:"recognize-error",id:m.id,error:String(e&&e.stack||e)})}
}
self.onmessage=e=>{
  const m=e.data;
  if(m.type==="init")init(m);
  else if(m.type==="recognize")recognize(m);
};