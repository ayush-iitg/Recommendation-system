# models/item_cf.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_pt_and_similarity(ratings_cf, min_ratings=20):
    """
    Build a smaller pivot table by keeping only
    books with enough ratings — reduces memory drastically
    """
    # Keep only books rated by 20+ users
    book_counts = ratings_cf["ISBN"].value_counts()
    popular     = book_counts[book_counts >= min_ratings].index
    filtered    = ratings_cf[ratings_cf["ISBN"].isin(popular)]

    pt = filtered.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating"
    ).fillna(0)

    # Use float32 instead of float64 — cuts memory in half
    pt_values = pt.values.astype("float32")

    similarity = cosine_similarity(pt_values)
    similarity = similarity.astype("float32")

    print(f"Pivot table : {pt.shape}")
    print(f"Similarity  : {similarity.shape}")
    print(f"Memory used : ~{pt_values.nbytes / 1e6:.0f}MB "
          f"(pivot) + "
          f"~{similarity.nbytes / 1e6:.0f}MB (similarity)")
    return pt, similarity


def item_cf_recommend(book_title, pt, similarity,
                      books, top_k=8):
    if book_title not in pt.index:
        return pd.DataFrame(columns=["Book-Title",
                                     "Book-Author",
                                     "similarity"])

    idx        = pt.index.tolist().index(book_title)
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores,
                        key=lambda x: x[1],
                        reverse=True)[1:top_k+1]

    similar_titles = [pt.index[i] for i, _ in sim_scores]
    scores         = [round(float(s), 4) for _, s in sim_scores]

    result = books[
        books["Book-Title"].isin(similar_titles)
    ][["Book-Title", "Book-Author", "Image-URL-M"]]\
     .drop_duplicates("Book-Title").copy()

    score_map            = dict(zip(similar_titles, scores))
    result["similarity"] = result["Book-Title"].map(score_map)
    return result.sort_values("similarity", ascending=False)