# models/svd_model.py
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split


def train_svd(ratings_cf):

    reader = Reader(rating_scale=(1, 10))
    data   = Dataset.load_from_df(
        ratings_cf[["User-ID", "ISBN", "Book-Rating"]],
        reader
    )

    trainset, testset = train_test_split(
        data, test_size=0.2, random_state=42)

    model = SVD(n_factors=50, n_epochs=20,
                lr_all=0.005, reg_all=0.02,
                random_state=42)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
    print(f"SVD → RMSE: {rmse:.4f}   MAE: {mae:.4f}")

    return model, trainset


def svd_predict(model, user_id, isbn):
    pred = model.predict(str(user_id), str(isbn))
    return float(np.clip(pred.est, 1, 10))


def svd_recommend(model, user_id, ratings_cf,
                  books, top_k=8):
    seen   = set(ratings_cf[
        ratings_cf["User-ID"] == user_id]["ISBN"])
    unseen = [isbn for isbn in books["ISBN"].unique()
              if isbn not in seen]

    preds = [(isbn, svd_predict(model, user_id, isbn))
             for isbn in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_k]

    top_isbns  = [p[0] for p in top]
    top_scores = [round(p[1], 3) for p in top]

    result = books[books["ISBN"].isin(top_isbns)][
        ["ISBN", "Book-Title", "Book-Author", "Image-URL-M"]
    ].drop_duplicates("ISBN").copy()

    score_map = dict(zip(top_isbns, top_scores))
    result["predicted_rating"] = result["ISBN"].map(score_map)
    return result.sort_values("predicted_rating",
                               ascending=False)


if __name__ == "__main__":
    books      = pd.read_csv("data/books_clean.csv")
    ratings_cf = pd.read_csv("data/ratings_cf.csv")
    model, _   = train_svd(ratings_cf)
    print(f"Training done")