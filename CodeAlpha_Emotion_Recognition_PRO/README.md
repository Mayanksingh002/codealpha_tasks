# 🎙️ EmoSense AI — Speech Emotion Recognition

Deep-learning speech emotion recognition system aligned with CodeAlpha Task 2.

### Highlights
- RAVDESS-compatible audio loader
- MFCC + delta + delta-delta features
- Speaker-independent split by actor
- CNN + BiLSTM + attention
- Class-weighted training
- Early model checkpointing
- Macro-F1 and confusion matrix evaluation
- Audio waveform/spectrogram-ready dashboard
- Prediction confidence visualization
- Responsible-use notes around uncertainty

### Run
```bash
pip install -r requirements.txt
python train.py --data data/ravdess
streamlit run app.py
```
