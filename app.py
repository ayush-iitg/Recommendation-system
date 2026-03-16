# app.py
import streamlit as st
import pandas as pd
import pickle
import os
from models.popularity    import weighted_popularity
from models.item_cf       import item_cf_recommend
from models.svd_model     import svd_recommend
from models.hybrid        import hybrid_recommend

st.set_page_config(page_title="Book Recommender",
                   page_icon="📚", layout="wide")
st.title("Smart Book Recommender")

# ── Check models exist ──────────────────────────────────────────────
models_exist = all(os.path.exists(f"models/{f}") for f in
                   ["pt.pkl", "sim.pkl", "svd.pkl"])

if not models_exist:
    st.error("Models not found. Run: python precompute.py")
    st.stop()

@st.cache_data
def load_data():
    books      = pd.read_csv("data/books_clean.csv")
    ratings_cf = pd.read_csv("data/ratings_cf.csv")
    return books, ratings_cf

@st.cache_resource
def load_models():
    print("Loading precomputed models from disk...")
    pt         = pickle.load(open("models/pt.pkl",  "rb"))
    sim        = pickle.load(open("models/sim.pkl", "rb"))
    svd        = pickle.load(open("models/svd.pkl", "rb"))
    print("All models loaded!")
    return pt, sim, svd

books, ratings_cf       = load_data()
pt, sim, svd_model = load_models()

tab1, tab2, tab3, tab4 = st.tabs([
    "Popular Books",
    "Search by Title",
    "For You",
    "New User"
])

def book_card(row):
    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            img = row.get("Image-URL-M", "")
            if pd.notna(img) and str(img) != "":
                st.image(str(img), width=80)
        with c2:
            st.markdown(f"**{row['Book-Title']}**")
            st.caption(f"{row['Book-Author']}")
            for key in ["weighted_score", "similarity",
                        "predicted_rating", "hybrid_score"]:
                if key in row and row[key] != "":
                    st.caption(f"Score: {row[key]}")
                    break
        st.divider()

# ── Tab 1: Popular Books ────────────────────────────────────────────
with tab1:
    st.subheader("Top rated books")
    popular = weighted_popularity(ratings_cf, books, top_k=20)
    for _, row in popular.iterrows():
        book_card(row)

# ── Tab 2: Search by Title ──────────────────────────────────────────
# ── Tab 2: Search by Title ──────────────────────────────────────────
with tab2:
    st.subheader("Find similar books")

    title_input = st.selectbox(
        "Select a book you like:",
        sorted(pt.index.tolist())
    )

    with st.spinner("Finding similar books..."):
        recs = item_cf_recommend(
            title_input, pt, sim, books)

    if recs.empty:
        st.info("No similar books found.")
    else:
        for _, row in recs.iterrows():
            book_card(row)

# ── Tab 3: For You ─────────────────────────────────────────────────
# ── Tab 3: For You ─────────────────────────────────────────────────
# ── Tab 3: For You ─────────────────────────────────────────────────
# ── Tab 3: For You ─────────────────────────────────────────────────
with tab3:
    st.subheader("Get personalized recommendations")

    all_users = sorted(ratings_cf["User-ID"].unique().tolist())
    user_id   = st.selectbox("Select your User ID",
                              options=all_users)

    with st.spinner("Finding books you'll love..."):
        recs = hybrid_recommend(
            user_id, ratings_cf, books,
            svd_model, pt, sim)

    if recs.empty:
        st.info("Not enough rating history for this user. "
                "Try another User ID.")
    else:
        st.success(f"Top picks for User {user_id}")
        for _, row in recs.iterrows():
            book_card(row)
# ── Tab 4: New User ─────────────────────────────────────────────────
with tab4:
    st.subheader("New here? Start with our top picks")
    popular = weighted_popularity(ratings_cf, books, top_k=10)
    for _, row in popular.iterrows():
        book_card(row)


