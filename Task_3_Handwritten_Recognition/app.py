import streamlit as st,torch,numpy as np
from PIL import Image,ImageOps
from train import Net
st.set_page_config(page_title="VisionWrite AI",page_icon="✍️")
st.title("✍️ VisionWrite AI")
st.caption("Handwritten digit recognition • CNN • Top-3 confidence")
f=st.file_uploader("Upload a handwritten digit image",type=["png","jpg","jpeg"])
if f:
 im=Image.open(f).convert("L");im=ImageOps.invert(im).resize((28,28));x=np.array(im,dtype=np.float32)/255;x=(x-.1307)/.3081
 m=Net();m.load_state_dict(torch.load("artifacts/mnist_cnn.pt",map_location="cpu"));m.eval()
 with torch.no_grad():p=torch.softmax(m(torch.tensor(x).unsqueeze(0).unsqueeze(0)),1)[0]
 vals,idx=torch.topk(p,3);st.image(im,width=180);st.metric("Prediction",str(int(idx[0])),f"{vals[0]:.1%} confidence")
 st.subheader("Top-3 predictions")
 st.bar_chart({str(int(idx[i])):float(vals[i]) for i in range(3)})
