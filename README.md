# 📚 Smart Book Recommender System

A hybrid book recommendation system built on the **Book-Crossing dataset** (1M+ ratings from 278K users across 271K books). Combines SVD matrix factorization and item-based collaborative filtering to deliver personalized book recommendations.

🔗 **Live Demo:** [adv-recommendation-system.streamlit.app](adv-book-recommender-app.streamlit.app)

---

## Features

- **Popular Books** — IMDB weighted rating formula ranks top books by combining average rating with number of ratings
- **Search by Title** — Item-based collaborative filtering finds books similar to one you already like
- **For You** — Hybrid model (SVD + item-CF) generates personalized recommendations based on your rating history
- **New User** — Popularity-based cold start for users with no history

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy |
| ML | scikit-surprise (SVD), scikit-learn (cosine similarity) |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Algorithms

### 1. Weighted Popularity (IMDB Formula)
```
WR = (v / (v+m)) × R + (m / (v+m)) × C

v = number of ratings for this book
R = average rating of this book
m = minimum ratings threshold (10)
C = global mean rating across all books
```
Prevents books with few ratings from unfairly dominating the rankings.

### 2. Item-Based Collaborative Filtering
Builds a user-item pivot table and computes cosine similarity between all book pairs. Recommends books that were rated similarly by the same users.

### 3. SVD Matrix Factorization
Decomposes the user-item ratings matrix into latent user and item factor matrices using scikit-surprise. Learns hidden taste dimensions from rating patterns to predict unseen ratings.

### 4. Hybrid Model
Combines SVD and item-CF scores with weighted averaging:
```
hybrid_score = 0.6 × SVD_score + 0.4 × ItemCF_score
```

---

## Dataset

**Book-Crossing Dataset** — [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

| File | Rows | Description |
|---|---|---|
| Books.csv | 271,360 | Title, author, publisher, cover image |
| Ratings.csv | 1,149,780 | User-book ratings (1-10) |
| Users.csv | 278,858 | Age, location |

After cleaning and filtering:
- Removed implicit ratings (rating = 0)
- Kept users with 50+ ratings
- Kept books with 10+ ratings
- Final CF dataset: ~11K ratings, ~1000 users, ~350 books
- Matrix sparsity: ~93%

---

## Project Structure
```
book-recommender/
├── data/
│   ├── books_clean.csv       ← cleaned books
│   ├── ratings_cf.csv        ← filtered ratings for CF
│   ├── ratings_clean.csv     ← all explicit ratings
│   └── users_clean.csv       ← cleaned users
├── models/
│   ├── hybrid.py             ← hybrid recommender
│   ├── item_cf.py            ← item-based CF
│   ├── popularity.py         ← IMDB weighted ranking
│   ├── svd_model.py          ← SVD matrix factorization
│   ├── pt.pkl                ← precomputed pivot table
│   ├── sim.pkl               ← precomputed similarity matrix
│   └── svd.pkl               ← trained SVD model
├── notebooks/
│   └── EDA.ipynb             ← exploratory data analysis
├── app.py                    ← Streamlit frontend
├── precompute.py             ← offline model training
└── requirements.txt
```

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/book-recommender.git
cd book-recommender
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) and place `Books.csv`, `Ratings.csv`, `Users.csv` in the project root.

### 5. Clean the data
```bash
python -c "from data.load_data import load_and_clean; load_and_clean()"
```

### 6. Precompute models
```bash
python precompute.py
```

### 7. Run the app
```bash
streamlit run app.py
```

---

## How It Works
```
Raw data (1.1M ratings)
        ↓
Remove zero ratings + filter active users/popular books
        ↓
11K ratings, 1000 users, 350 books
        ↓
┌─────────────────┬──────────────────────┐
│  Pivot table    │   SVD training       │
│  + cosine sim   │   (10 epochs SGD)    │
└────────┬────────┴──────────┬───────────┘
         │                   │
         └─────────┬─────────┘
                   ↓
            Hybrid scorer
       0.6 × SVD + 0.4 × ItemCF
                   ↓
         Streamlit UI (4 tabs)
```
