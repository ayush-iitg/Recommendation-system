# models/popularity.py
import pandas as pd
import numpy as np


def weighted_popularity(ratings_cf, books, top_k=50):
    """
    IMDB Weighted Rating Formula:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    v = number of ratings for this book
    m = minimum ratings required
    R = mean rating for this book
    C = global mean rating across all books
    """
    C = ratings_cf["Book-Rating"].mean()
    m = 10  # minimum ratings threshold

    # Aggregate ratings per book
    book_stats = (ratings_cf
                  .groupby("ISBN")["Book-Rating"]
                  .agg(["count", "mean"])
                  .rename(columns={"count": "num_ratings",
                                   "mean":  "avg_rating"})
                  .reset_index())

    # Only books with enough ratings
    book_stats = book_stats[book_stats["num_ratings"] >= m]

    # Apply IMDB formula
    v = book_stats["num_ratings"]
    R = book_stats["avg_rating"]
    book_stats["weighted_score"] = (
        (v / (v + m)) * R + (m / (v + m)) * C
    )

    # Merge with book metadata
    # Check which columns are available
    meta_cols = ["ISBN", "Book-Title", "Book-Author", "Image-URL-M"]
    if "reading_difficulty" in books.columns:
        meta_cols.append("reading_difficulty")

    popular = (book_stats
               .sort_values("weighted_score", ascending=False)
               .head(top_k)
               .merge(books[meta_cols].drop_duplicates("ISBN"),
                      on="ISBN", how="left"))

    popular["weighted_score"] = popular["weighted_score"].round(3)
    popular["avg_rating"]     = popular["avg_rating"].round(2)
    return popular


# def popular_by_difficulty(ratings_cf, books, difficulty, top_k=10):
#     """
#     Cold start: new user picks difficulty level
#     → return popular books at that reading level
#     """
#     if "reading_difficulty" not in books.columns:
#         # fallback if column missing
#         return weighted_popularity(ratings_cf, books, top_k=top_k)

#     diff_isbns = books[
#         books["reading_difficulty"] == difficulty
#     ]["ISBN"]

#     filtered = ratings_cf[ratings_cf["ISBN"].isin(diff_isbns)]

#     if filtered.empty:
#         # fallback to all books if no books found at this level
#         print(f"No books found for difficulty '{difficulty}', "
#               f"returning overall popular books")
#         return weighted_popularity(ratings_cf, books, top_k=top_k)

#     return weighted_popularity(filtered, books, top_k=top_k)


if __name__ == "__main__":
    # Quick test
    books      = pd.read_csv("data/books_clean.csv")
    ratings_cf = pd.read_csv("data/ratings_cf.csv")

    print("Testing weighted popularity...")
    popular = weighted_popularity(ratings_cf, books, top_k=10)
    print(popular[["Book-Title", "Book-Author",
                   "num_ratings", "avg_rating",
                   "weighted_score"]].to_string())

    # print("\nTesting popular by difficulty (Easy)...")
    # easy = popular_by_difficulty(ratings_cf, books, "Easy", top_k=5)
    # print(easy[["Book-Title", "weighted_score"]].to_string())
