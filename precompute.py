# precompute.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from models.item_cf   import build_pt_and_similarity
from models.svd_model import train_svd

os.makedirs("models", exist_ok=True)

# ── Load cleaned data ──────────────────────────────────────────────
print("Loading data...")
books      = pd.read_csv("data/books_clean.csv")
ratings_cf = pd.read_csv("data/ratings_cf.csv")
print(f"  Books:      {books.shape}")
print(f"  Ratings CF: {ratings_cf.shape}")

# ── Item-CF pivot table + similarity ──────────────────────────────
print("\nBuilding item-CF pivot table + similarity matrix...")
pt, sim = build_pt_and_similarity(ratings_cf)
pickle.dump(pt,  open("models/pt.pkl",  "wb"), protocol=4)
pickle.dump(sim, open("models/sim.pkl", "wb"), protocol=4)
print("Saved: models/pt.pkl  models/sim.pkl")

# ── SVD model ─────────────────────────────────────────────────────
# Training SVD section 

print("\nTraining SVD model...")
svd_model, trainset = train_svd(ratings_cf)
pickle.dump(svd_model, open("models/svd.pkl", "wb"), protocol=4)
print("Saved: models/svd.pkl")
# ── Verify all files ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("ALL MODELS SAVED")
print("=" * 50)
for f in ["pt.pkl", "sim.pkl", "svd.pkl",
          "c_sim.pkl", "books_content.pkl"]:
    path = f"models/{f}"
    size = os.path.getsize(path) / 1e6
    print(f"  {path}  ({size:.1f} MB)")

print("\nDone! Run: streamlit run app.py")