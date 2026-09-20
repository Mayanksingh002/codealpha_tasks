import os,torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
os.makedirs("artifacts",exist_ok=True);dev="cuda" if torch.cuda.is_available() else "cpu"
trf=transforms.Compose([transforms.RandomAffine(12,translate=(.08,.08),scale=(.9,1.1)),transforms.ToTensor(),transforms.Normalize((.1307,),(.3081,))])
tef=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.1307,),(.3081,))])
tr=datasets.MNIST("data",True,download=True,transform=trf);te=datasets.MNIST("data",False,download=True,transform=tef)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.n=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(.15),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(.2),nn.Flatten(),nn.Linear(128*7*7,256),nn.ReLU(),nn.Dropout(.35),nn.Linear(256,10))
 def forward(self,x):return self.n(x)
m=Net().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,15);loss=nn.CrossEntropyLoss(label_smoothing=.05);best=0
for e in range(15):
 m.train()
 for x,y in DataLoader(tr,128,shuffle=True):x,y=x.to(dev),y.to(dev);opt.zero_grad();loss(m(x),y).backward();opt.step()
 m.eval();co=to=0
 with torch.no_grad():
  for x,y in DataLoader(te,256):co+=(m(x.to(dev)).argmax(1)==y.to(dev)).sum().item();to+=len(y)
 acc=co/to;sch.step();print(e+1,acc)
 if acc>best:best=acc;torch.save(m.state_dict(),"artifacts/mnist_cnn.pt")
print("Best:",best)
