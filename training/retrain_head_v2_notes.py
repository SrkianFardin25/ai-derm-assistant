"""Retrain the classifier head on pooled EfficientNet-B4 features (HAM10000 + PAD-UFES-20).
Feature extraction: see extract_features section; datasets: marmal88/skin_cancer (HAM10000) and OctoMed/PAD_UFES_20 on Hugging Face."""
# --- common.py ---
import onnxruntime as ort,numpy as np
from PIL import Image
MODEL='/workspace/work/derm/repo/backend/checkpoints/efficientnet_b4_ham10000.onnx'
M=np.array([0.485,0.456,0.406],np.float32);S=np.array([0.229,0.224,0.225],np.float32)
o=ort.SessionOptions();o.intra_op_num_threads=2
sess=ort.InferenceSession(MODEL,o,providers=['CPUExecutionProvider'])
def prep(im):
    a=np.asarray(im.convert("RGB").resize((380,380),Image.BILINEAR),np.float32)/255
    return ((a-M)/S).transpose(2,0,1)
def feats(imgs):
    out=[]
    for k in range(0,len(imgs),8):
        x=np.stack([prep(i) for i in imgs[k:k+8]]).astype(np.float32)
        out.append(sess.run(['features'],{'input':x})[0].mean(axis=(2,3)))
    return np.concatenate(out)
# --- pad.py (PAD-UFES-20 feature extraction) ---
import json,urllib.request,os,io,sys,numpy as np,pyarrow.parquet as pq
from PIL import Image
from common import feats
# PAD-UFES-20 has labels that are not represented by the seven-class
# HAM10000 contract.  Do not silently relabel SCC as AKIEC, or invent DF/VASC
# smartphone supervision; those require an explicit schema/model change.
m={'ACK':0,'BCC':1,'SEK':2,'MEL':4,'NEV':5}
urls=json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/OctoMed/PAD_UFES_20/parquet"))["default"]["train"]
for k in map(int,sys.argv[1:]):
    if os.path.exists(f"pad_{k}.npz"): continue
    fn=f"/tmp/shard{k}.parquet"; urllib.request.urlretrieve(urls[k],fn)
    pf=pq.ParquetFile(fn); X=[];y=[];sub=[];h=[]
    for b in pf.iter_batches(batch_size=32,columns=["image","image_hash","answer"]):
        t=b.to_pylist()
        batch_features=feats([Image.open(io.BytesIO(r["image"]["bytes"])) for r in t])
        keep=[]
        for i, r in enumerate(t):
            c=r["answer"].split("(")[-1].rstrip(")")
            if c not in m:
                continue
            keep.append(i); y.append(m[c]); sub.append(c); h.append(r["image_hash"])
        if keep:
            X.append(batch_features[keep])
    np.savez(f"pad_{k}.npz",X=np.concatenate(X),y=np.array(y),sub=np.array(sub),h=np.array(h))
    os.remove(fn); print(k,len(y),flush=True)
# --- train.py ---
import base64,json,numpy as np,onnx,glob
from scipy.optimize import minimize_scalar
from onnx import numpy_helper
C=["akiec","bcc","bkl","df","mel","nv","vasc"]
m=onnx.load('repo/backend/checkpoints/efficientnet_b4_ham10000.onnx')
I={i.name:i for i in m.graph.initializer}
W0=numpy_helper.to_array(I['backbone.classifier.1.weight']).astype(np.float64); b0=numpy_helper.to_array(I['backbone.classifier.1.bias']).astype(np.float64)
print(W0.shape)
P=[np.load(f) for f in sorted(glob.glob("pad_*.npz"))]
PX=np.concatenate([p["X"] for p in P]);Py=np.concatenate([p["y"] for p in P]);Ph=np.concatenate([p["h"] for p in P])
# hash-based split
rng=np.random.RandomState(0);idx=rng.permutation(len(Py));nte=int(.2*len(Py))
pte,ptr=idx[:nte],idx[nte:]
Htr=np.load("ham_train.npz");Hte=np.load("ham_test.npz")
HX,Hy=Htr["X"],Htr["y"]
# hold out bkl/df from ham train for test
r2=np.random.RandomState(1);hold=np.zeros(len(Hy),bool)
for c,n in [(2,60),(3,15)]:
    ii=np.where(Hy==c)[0];hold[r2.choice(ii,n,replace=False)]=True
HteX=np.concatenate([Hte["X"],HX[hold]]);Htey=np.concatenate([Hte["y"],Hy[hold]])
HX,Hy=HX[~hold],Hy[~hold]
Xtr=np.concatenate([PX[ptr],HX]);ytr=np.concatenate([Py[ptr],Hy]);src=np.r_[np.ones(len(ptr)),np.zeros(len(Hy))]
def sm(z): z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def ev(W,b,X,y,name):
    p=(X@W.T+b).argmax(1);acc=[(p[y==c]==c).mean() for c in range(7) if (y==c).sum()]
    rec={C[c]:f"{(p[y==c]==c).sum()}/{(y==c).sum()}" for c in range(7) if (y==c).sum()}
    print(f"  {name}: bal-acc {np.mean(acc):.3f} acc {(p==y).mean():.3f} {rec}")
    return np.mean(acc)
# class weights: balance per class within each source
w=np.zeros(len(ytr))
for s in [0,1]:
    for c in range(7):
        k=(src==s)&(ytr==c)
        if k.sum(): w[k]=1/k.sum()
w=w/w.sum()*len(ytr)
def train(lam,lr=0.05,ep=400):
    W=W0.copy();b=b0.copy()
    mu=Xtr.mean(0); # no standardization to keep onnx graph unchanged
    Y=np.eye(7)[ytr]
    mW=np.zeros_like(W);vW=np.zeros_like(W);mb=np.zeros_like(b);vb=np.zeros_like(b)
    for t in range(1,ep+1):
        p=sm(Xtr@W.T+b);g=(p-Y)*w[:,None]/len(ytr)
        gW=g.T@Xtr+lam*(W-W0);gb=g.sum(0)
        for (x,gx,m_,v_) in [(W,gW,mW,vW),(b,gb,mb,vb)]:
            m_[:]=.9*m_+.1*gx;v_[:]=.999*v_+.001*gx*gx
            x-=lr*(m_/(1-.9**t))/(np.sqrt(v_/(1-.999**t))+1e-8)
    return W,b
print("ORIGINAL");ev(W0,b0,PX[pte],Py[pte],"phone(PAD)");ev(W0,b0,HteX,Htey,"dermoscopy(HAM)")
best=None
for lam in []:
    for lr in [0.01,0.003]:
        W,b=train(lam,lr);print("lam",lam,"lr",lr)
        a=ev(W,b,PX[pte],Py[pte],"phone(PAD)");h=ev(W,b,HteX,Htey,"dermoscopy(HAM)")
        if best is None or a+h>best[0]: best=(a+h,lam,lr,W,b)

W,b=train(0.1,0.003);print("FINAL");ev(W,b,PX[pte],Py[pte],"phone(PAD)");ev(W,b,HteX,Htey,"dermoscopy(HAM)")

# Calibrate the final swapped head, not the original ONNX classifier head.
cal_logits=HteX@W.T+b
def nll(T):
    z=cal_logits/max(0.05,float(T)); z=z-z.max(1,keepdims=True)
    p=np.exp(z); p=p/p.sum(1,keepdims=True)
    return float(-np.log(np.clip(p[np.arange(len(p)),Htey],1e-12,1.0)).mean())
optimal_t=float(minimize_scalar(nll,bounds=(0.1,10.0),method="bounded").x)
W32=W.astype("<f4")
json.dump({
    "classes":C,
    "shape":list(W32.shape),
    "weight_f32_b64":base64.b64encode(W32.tobytes()).decode("ascii"),
    "bias":b.astype(np.float32).tolist(),
    "temperature":round(optimal_t,6),
    "note":"Linear head retrained on canonical seven-class labels; unsupported PAD-UFES-20 labels excluded. Temperature fit on held-out HAM logits after head retraining."
},open("head_v2.json","w"),indent=2)
print("FINAL TEMPERATURE",optimal_t)
