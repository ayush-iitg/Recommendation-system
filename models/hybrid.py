# models/hybrid.py
import pandas as pd
import numpy as np


def hybrid_recommend(user_id, ratings_cf, books,
                     svd_model, pt, similarity,
                     top_k=8, w_svd=0.6, w_cf=0.4):

    # ── Known books from ratings_cf only ──────────────────────────
    known_isbns = ratings_cf["ISBN"].unique().tolist()
    seen        = set(ratings_cf[
        ratings_cf["User-ID"] == user_id]["ISBN"])
    unseen      = [isbn for isbn in known_isbns
                   if isbn not in seen]

    if not unseen:
        return books[books["ISBN"].isin(known_isbns)].head(top_k)

    # ── SVD scores using surprise model ───────────────────────────
    svd_scores = {
        isbn: float(np.clip(
            svd_model.predict(str(user_id), str(isbn)).est,
            1, 10))
        for isbn in unseen
    }

    # ── Item-CF scores ─────────────────────────────────────────────
    user_ratings = (ratings_cf[ratings_cf["User-ID"] == user_id]
                    .sort_values("Book-Rating", ascending=False)
                    .head(5))

    cf_scores = {}
    pt_titles = pt.index.tolist()

    for isbn in unseen:
        title_match = books[
            books["ISBN"] == isbn]["Book-Title"].values
        if len(title_match) == 0 or \
           title_match[0] not in pt_titles:
            cf_scores[isbn] = 0.0
            continue

        title = title_match[0]
        t_idx = pt_titles.index(title)
        score = 0.0
        count = 0

        for _, row in user_ratings.iterrows():
            anchor = books[
                books["ISBN"] == row["ISBN"]
            ]["Book-Title"].values
            if len(anchor) == 0 or \
               anchor[0] not in pt_titles:
                continue
            a_idx  = pt_titles.index(anchor[0])
            score += float(similarity[a_idx][t_idx]) \
                     * row["Book-Rating"]
            count += 1

        cf_scores[isbn] = score / count if count > 0 else 0.0

    # ── Normalize CF scores to 1-10 ────────────────────────────────
    cf_vals = np.array(list(cf_scores.values()))
    if cf_vals.max() > 0:
        cf_vals = 1 + 9 * (cf_vals / cf_vals.max())
    cf_norm = dict(zip(cf_scores.keys(), cf_vals))

    # ── Combine scores ─────────────────────────────────────────────
    results = []
    for isbn in unseen:
        svd_s    = svd_scores.get(isbn, 0.0)
        cf_s     = cf_norm.get(isbn, 0.0)
        hybrid_s = w_svd * svd_s + w_cf * cf_s
        results.append((isbn, hybrid_s))

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:top_k]

    top_isbns = [r[0] for r in top]
    result    = books[books["ISBN"].isin(top_isbns)][
        ["ISBN", "Book-Title", "Book-Author", "Image-URL-M"]
    ].drop_duplicates("ISBN").copy()

    score_map = {r[0]: round(r[1], 3) for r in top}
    result["hybrid_score"] = result["ISBN"].map(score_map)
    return result.sort_values("hybrid_score", ascending=False)