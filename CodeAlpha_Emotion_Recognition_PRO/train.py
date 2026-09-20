import argparse,os
from pathlib import Path
import numpy as np,torch,librosa
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
EMO={1:"neutral",2:"calm",3:"happy",4:"sad",5:"angry",6:"fearful",7:"disgust",8:"surprised"}
def parse(p): 
 n=Path(p).stem.split("-"); return int(n[6]),EMO[int(n[2])]
def feat(p,sr=16000,seconds=3):
 y,_=librosa.load(p,sr=sr,mono=True); t=sr*seconds; y=y[:t]
 if len(y)<t:y=np.pad(y,(0,t-len(y)))
 m=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40,n_fft=1024,hop_length=256)
 x=np.concatenate([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)],0).T
 return ((x-x.mean(0))/(x.std(0)+1e-6)).astype(np.float32)
class DS(Dataset):
 def __init__(self,a):self.a=a
 def __len__(self):return len(self.a)
 def __getitem__(self,i):p,y=self.a[i];return torch.tensor(feat(p)),torch.tensor(y)
class CRNN(nn.Module):
 def __init__(self,n):
  super().__init__();self.c=nn.Sequential(nn.Conv1d(120,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(128,192,5,padding=2),nn.BatchNorm1d(192),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(192,256,3,padding=1),nn.ReLU())
  self.r=nn.LSTM(256,128,batch_first=True,bidirectional=True);self.a=nn.Linear(256,1);self.f=nn.Sequential(nn.Dropout(.35),nn.Linear(256,n))
 def forward(self,x):
  x=self.c(x.transpose(1,2)).transpose(1,2);x,_=self.r(x);w=torch.softmax(self.a(x).squeeze(-1),1).unsqueeze(-1);return self.f((x*w).sum(1))
def main(root):
 files=list(Path(root).rglob("*.wav"))
 if not files:raise SystemExit("No .wav files found.")
 raw=[(str(p),parse(p)[1]) for p in files]; labels=sorted(set(y for _,y in raw)); enc={e:i for i,e in enumerate(labels)}; items=[(p,enc[y]) for p,y in raw]
 actors=sorted(set(parse(p)[0] for p,_ in items));rng=np.random.default_rng(42);rng.shuffle(actors);n=len(actors)
 A=set(actors[:int(.7*n)]);B=set(actors[int(.7*n):int(.85*n)])
 tr=[x for x in items if parse(x[0])[0] in A];va=[x for x in items if parse(x[0])[0] in B]
 dev="cuda" if torch.cuda.is_available() else "cpu";m=CRNN(len(labels)).to(dev)
 w=compute_class_weight("balanced",classes=np.arange(len(labels)),y=[y for _,y in tr]);loss=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=dev));opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
 best=-1;os.makedirs("artifacts",exist_ok=True)
 for ep in range(30):
  m.train()
  for x,y in DataLoader(DS(tr),16,shuffle=True):x,y=x.to(dev),y.to(dev);opt.zero_grad();loss(m(x),y).backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  m.eval();P=[];Y=[]
  with torch.no_grad():
   for x,y in DataLoader(DS(va),16):P.extend(m(x.to(dev)).argmax(1).cpu().numpy());Y.extend(y.numpy())
  f=f1_score(Y,P,average="macro");print(ep+1,f)
  if f>best:best=f;torch.save({"state":m.state_dict(),"labels":labels},"artifacts/emotion_crnn.pt")
if __name__=="__main__":a=argparse.ArgumentParser();a.add_argument("--data",required=True);main(a.parse_args().data)
