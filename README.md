# fast-face-retrieval

Setup (macOS)
```bash
git clone https://github.com/nnwetzel/fast-face-retrieval.git
cd fast-face-retrieval
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run feature extraction
```bash
python -m src.feature_extraction \
  --deeplake-uri hub://activeloop/lfw \
  --output data/embeddings/lfw_deeplake_embeddings.npz \
  --device cpu
```

Note: the output .npz is compressed and contains arrays: embeddings (N×2048),