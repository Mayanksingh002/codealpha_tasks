import streamlit as st,torch,tempfile,librosa,plotly.graph_objects as go
from train import CRNN,feat
st.set_page_config(page_title="EmoSense AI",page_icon="🎙️",layout="wide")
st.title("🎙️ EmoSense AI")
st.caption("Speech Emotion Recognition • CNN + BiLSTM + Attention")
f=st.file_uploader("Upload a WAV file",type=["wav"])
if f:
 with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as t:t.write(f.read());p=t.name
 ck=torch.load("artifacts/emotion_crnn.pt",map_location="cpu");m=CRNN(len(ck["labels"]));m.load_state_dict(ck["state"]);m.eval()
 x=torch.tensor(feat(p)).unsqueeze(0)
 with torch.no_grad():probs=torch.softmax(m(x),1)[0].numpy()
 i=int(probs.argmax())
 c1,c2=st.columns([1,2]);c1.metric("Predicted emotion",ck["labels"][i]);c1.metric("Confidence",f"{probs[i]:.1%}")
 fig=go.Figure(go.Bar(x=ck["labels"],y=probs));fig.update_layout(title="Emotion probability profile",yaxis_title="Probability")
 c2.plotly_chart(fig,use_container_width=True)
 st.audio(f)
 st.info("Confidence is model probability, not a guarantee of the speaker's actual emotional state.")
