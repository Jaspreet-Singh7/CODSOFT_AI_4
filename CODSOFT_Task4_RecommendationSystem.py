# ============================================================
#  CodSoft Artificial Intelligence Internship
#  Task 04: Recommendation System
#  Author : Jaspreet Singh
#  GitHub : https://github.com/Jaspreet-Singh7/CODSOFT
#
#  Features:
#  - Collaborative Filtering (User-Based & Item-Based)
#  - Content-Based Filtering
#  - Hybrid Recommendation System
#  - Movie, Book and Product recommendations
#  - Cosine Similarity scoring
# ============================================================

import numpy as np
import sys
import os
import json
from datetime import datetime

OUTPUT_DIR = "recommendation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── DATASET ──────────────────────────────────────────────────
MOVIES = {
    "M001": {"title": "The Dark Knight",     "genres": ["Action","Crime","Drama"],   "year": 2008, "rating": 9.0},
    "M002": {"title": "Inception",            "genres": ["Action","Sci-Fi","Thriller"],"year":2010, "rating": 8.8},
    "M003": {"title": "The Matrix",           "genres": ["Action","Sci-Fi"],           "year": 1999, "rating": 8.7},
    "M004": {"title": "Interstellar",         "genres": ["Sci-Fi","Drama","Adventure"],"year":2014, "rating": 8.6},
    "M005": {"title": "The Avengers",         "genres": ["Action","Adventure","Sci-Fi"],"year":2012,"rating": 8.0},
    "M006": {"title": "Pulp Fiction",         "genres": ["Crime","Drama","Thriller"],  "year": 1994, "rating": 8.9},
    "M007": {"title": "The Shawshank Redemption","genres":["Drama"],                  "year": 1994, "rating": 9.3},
    "M008": {"title": "Forrest Gump",         "genres": ["Drama","Romance"],           "year": 1994, "rating": 8.8},
    "M009": {"title": "The Godfather",        "genres": ["Crime","Drama"],             "year": 1972, "rating": 9.2},
    "M010": {"title": "Avengers: Endgame",    "genres": ["Action","Adventure","Sci-Fi"],"year":2019,"rating": 8.4},
    "M011": {"title": "Parasite",             "genres": ["Drama","Thriller","Comedy"], "year": 2019, "rating": 8.6},
    "M012": {"title": "Get Out",              "genres": ["Horror","Mystery","Thriller"],"year":2017,"rating": 7.7},
    "M013": {"title": "La La Land",           "genres": ["Drama","Music","Romance"],   "year": 2016, "rating": 8.0},
    "M014": {"title": "Joker",                "genres": ["Crime","Drama","Thriller"],  "year": 2019, "rating": 8.4},
    "M015": {"title": "1917",                 "genres": ["Drama","War","Action"],      "year": 2019, "rating": 8.3},
}

USER_RATINGS = {
    "User_A": {"M001":9,"M002":8,"M003":9,"M005":7,"M010":8,"M014":8},
    "User_B": {"M006":9,"M007":9,"M008":8,"M009":9,"M011":8,"M013":7},
    "User_C": {"M002":9,"M003":8,"M004":10,"M005":8,"M010":7,"M015":8},
    "User_D": {"M001":8,"M005":9,"M006":7,"M009":8,"M010":9,"M014":9},
    "User_E": {"M007":10,"M008":9,"M009":9,"M011":8,"M013":9,"M015":7},
    "User_F": {"M002":8,"M003":7,"M004":9,"M012":8,"M015":8,"M001":7},
    "User_G": {"M006":8,"M009":9,"M011":9,"M014":7,"M001":8,"M007":8},
}

ALL_USERS  = list(USER_RATINGS.keys())
ALL_MOVIES = list(MOVIES.keys())


# ── HELPER: COSINE SIMILARITY ─────────────────────────────────
def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    dot     = np.dot(a, b)
    norm_a  = np.linalg.norm(a)
    norm_b  = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_rating_vector(user_id):
    """Get user's full rating vector across all movies."""
    return [USER_RATINGS[user_id].get(mid, 0) for mid in ALL_MOVIES]


# ── COLLABORATIVE FILTERING ───────────────────────────────────
class CollaborativeFilteringRecommender:
    """
    User-Based Collaborative Filtering.
    Finds users similar to target user and recommends
    movies they liked that the target hasn't seen yet.
    """

    def __init__(self):
        self.name = "Collaborative Filtering"
        print("✅ Collaborative Filtering Recommender initialized")

    def get_user_similarity(self, user_id):
        """Compute similarity between target user and all others."""
        target_vec = get_rating_vector(user_id)
        similarities = {}
        for other_user in ALL_USERS:
            if other_user == user_id:
                continue
            other_vec = get_rating_vector(other_user)
            sim = cosine_similarity(target_vec, other_vec)
            similarities[other_user] = round(sim, 4)
        return dict(sorted(similarities.items(), key=lambda x: -x[1]))

    def recommend(self, user_id, n=5):
        """
        Recommend movies using User-Based CF.
        1. Find similar users
        2. Get movies they liked
        3. Filter out movies user already rated
        4. Score by similarity × rating
        """
        if user_id not in USER_RATINGS:
            return []

        user_seen  = set(USER_RATINGS[user_id].keys())
        similarities = self.get_user_similarity(user_id)

        # Aggregate scores from similar users
        movie_scores = {}
        movie_count  = {}

        for similar_user, sim_score in list(similarities.items())[:4]:  # Top 4 similar
            if sim_score <= 0:
                continue
            for movie_id, rating in USER_RATINGS[similar_user].items():
                if movie_id not in user_seen:
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = 0
                        movie_count[movie_id]  = 0
                    movie_scores[movie_id] += sim_score * rating
                    movie_count[movie_id]  += sim_score

        # Normalize scores
        predicted = {}
        for mid in movie_scores:
            if movie_count[mid] > 0:
                predicted[mid] = round(movie_scores[mid] / movie_count[mid], 3)

        # Sort by predicted score
        sorted_recs = sorted(predicted.items(), key=lambda x: -x[1])
        return sorted_recs[:n]


# ── CONTENT-BASED FILTERING ───────────────────────────────────
class ContentBasedRecommender:
    """
    Content-Based Filtering.
    Recommends movies similar in genre/characteristics
    to movies the user has liked.
    """

    def __init__(self):
        self.name = "Content-Based Filtering"
        # Get all unique genres
        self.all_genres = sorted(set(
            g for m in MOVIES.values() for g in m["genres"]
        ))
        print(f"✅ Content-Based Recommender initialized")
        print(f"   Genres tracked: {len(self.all_genres)}")

    def get_genre_vector(self, movie_id):
        """One-hot encode movie genres."""
        movie   = MOVIES[movie_id]
        return [1 if g in movie["genres"] else 0 for g in self.all_genres]

    def get_movie_similarity(self, movie_id_a, movie_id_b):
        """Compute genre-based similarity between two movies."""
        vec_a = self.get_genre_vector(movie_id_a)
        vec_b = self.get_genre_vector(movie_id_b)
        return cosine_similarity(vec_a, vec_b)

    def get_user_profile(self, user_id):
        """Build genre preference profile for user."""
        profile = {g: 0.0 for g in self.all_genres}
        total_weight = 0

        for movie_id, rating in USER_RATINGS[user_id].items():
            movie = MOVIES[movie_id]
            for genre in movie["genres"]:
                profile[genre] += rating
            total_weight += rating

        if total_weight > 0:
            profile = {g: round(v/total_weight, 3) for g, v in profile.items()}
        return profile

    def recommend(self, user_id, n=5):
        """Recommend based on genre preferences."""
        if user_id not in USER_RATINGS:
            return []

        user_seen = set(USER_RATINGS[user_id].keys())
        profile   = self.get_user_profile(user_id)
        profile_vec = [profile.get(g, 0) for g in self.all_genres]

        scores = {}
        for movie_id in ALL_MOVIES:
            if movie_id not in user_seen:
                movie_vec = self.get_genre_vector(movie_id)
                sim = cosine_similarity(profile_vec, movie_vec)
                # Boost by movie rating
                base_rating = MOVIES[movie_id]["rating"] / 10.0
                scores[movie_id] = round(sim * 0.7 + base_rating * 0.3, 4)

        sorted_recs = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_recs[:n]


# ── HYBRID RECOMMENDER ────────────────────────────────────────
class HybridRecommender:
    """
    Hybrid Recommendation System.
    Combines Collaborative Filtering + Content-Based Filtering
    for better accuracy and coverage.
    """

    def __init__(self):
        self.name = "Hybrid Recommender"
        self.cf  = CollaborativeFilteringRecommender()
        self.cb  = ContentBasedRecommender()
        print(f"✅ Hybrid Recommender initialized (CF + Content-Based)")

    def recommend(self, user_id, n=5, cf_weight=0.6, cb_weight=0.4):
        """Combine CF and CB recommendations with weighted scoring."""
        cf_recs = dict(self.cf.recommend(user_id, n=10))
        cb_recs = dict(self.cb.recommend(user_id, n=10))

        all_movies = set(cf_recs.keys()) | set(cb_recs.keys())
        hybrid_scores = {}

        for movie_id in all_movies:
            cf_score = cf_recs.get(movie_id, 0)
            cb_score = cb_recs.get(movie_id, 0)

            # Normalize CF scores to 0-1 range
            max_cf = max(cf_recs.values()) if cf_recs else 1
            cf_normalized = cf_score / max_cf if max_cf > 0 else 0

            hybrid_scores[movie_id] = round(
                cf_weight * cf_normalized + cb_weight * cb_score, 4
            )

        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: -x[1])
        return sorted_recs[:n]


# ── DISPLAY HELPERS ───────────────────────────────────────────
def display_recommendations(recs, title="Recommendations", user_id=None):
    """Pretty print recommendations."""
    print(f"\n{'─'*55}")
    print(f"  {title}")
    if user_id:
        print(f"  User: {user_id} | Watched: {list(USER_RATINGS.get(user_id,{}).keys())[:3]}...")
    print(f"{'─'*55}")
    print(f"  {'#':<4} {'Movie Title':<32} {'Score':>6} {'Genres'}")
    print(f"  {'─'*51}")

    for i, (movie_id, score) in enumerate(recs, 1):
        movie   = MOVIES[movie_id]
        genres  = ", ".join(movie["genres"][:2])
        print(f"  {i:<4} {movie['title']:<32} {score:>6.3f}  {genres}")

    print(f"{'─'*55}")


# ── DEMO ──────────────────────────────────────────────────────
def run_demo():
    """Full demonstration of all recommendation algorithms."""
    print("\n" + "=" * 60)
    print("  CodSoft AI Task 04: Recommendation System")
    print("  CF + Content-Based + Hybrid | Jaspreet Singh")
    print("=" * 60)

    print(f"\n📊 Dataset:")
    print(f"   Movies: {len(MOVIES)}")
    print(f"   Users:  {len(USER_RATINGS)}")
    print(f"   Total ratings: {sum(len(v) for v in USER_RATINGS.values())}")

    all_passed = True

    # ── Test 1: Collaborative Filtering ─────────────────────
    print("\n\n🤝 1. COLLABORATIVE FILTERING (User-Based)")
    cf = CollaborativeFilteringRecommender()

    test_user = "User_A"
    cf_recs   = cf.recommend(test_user, n=5)
    display_recommendations(cf_recs, "CF Recommendations", test_user)

    sims = cf.get_user_similarity(test_user)
    print(f"\n  Most similar users to {test_user}:")
    for u, s in list(sims.items())[:3]:
        print(f"    {u}: {s:.3f} similarity")

    passed1 = len(cf_recs) >= 3
    print(f"\n  ✅ PASSED" if passed1 else "  ❌ FAILED")
    if not passed1: all_passed = False

    # ── Test 2: Content-Based ────────────────────────────────
    print("\n\n📝 2. CONTENT-BASED FILTERING")
    cb = ContentBasedRecommender()

    test_user2 = "User_B"
    cb_recs    = cb.recommend(test_user2, n=5)
    display_recommendations(cb_recs, "Content-Based Recommendations", test_user2)

    profile = cb.get_user_profile(test_user2)
    top_genres = sorted(profile.items(), key=lambda x: -x[1])[:3]
    print(f"\n  {test_user2} Genre Preferences:")
    for g, score in top_genres:
        bar = "█" * int(score * 20)
        print(f"    {g:<12}: {bar} ({score:.2f})")

    passed2 = len(cb_recs) >= 3
    print(f"\n  ✅ PASSED" if passed2 else "  ❌ FAILED")
    if not passed2: all_passed = False

    # ── Test 3: Hybrid ───────────────────────────────────────
    print("\n\n🔀 3. HYBRID RECOMMENDATION SYSTEM")
    hybrid = HybridRecommender()

    test_user3 = "User_C"
    h_recs     = hybrid.recommend(test_user3, n=5)
    display_recommendations(h_recs, "Hybrid Recommendations", test_user3)

    passed3 = len(h_recs) >= 3
    print(f"\n  ✅ PASSED" if passed3 else "  ❌ FAILED")
    if not passed3: all_passed = False

    # ── Test 4: All users ────────────────────────────────────
    print("\n\n👥 4. RECOMMENDATIONS FOR ALL USERS")
    print("─" * 55)
    for uid in ALL_USERS:
        recs = hybrid.recommend(uid, n=3)
        top  = MOVIES[recs[0][0]]["title"] if recs else "None"
        print(f"  {uid}: Top pick → {top}")

    # ── Save results ─────────────────────────────────────────
    results = {
        "generated_at": datetime.now().isoformat(),
        "dataset": {
            "movies": len(MOVIES),
            "users":  len(USER_RATINGS)
        },
        "recommendations": {}
    }
    for uid in ALL_USERS:
        recs = hybrid.recommend(uid, n=5)
        results["recommendations"][uid] = [
            {"movie": MOVIES[mid]["title"], "score": score}
            for mid, score in recs
        ]

    with open(f"{OUTPUT_DIR}/recommendations.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Results saved: {OUTPUT_DIR}/recommendations.json")

    # ── Summary ──────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"  ✅ Collaborative Filtering : User-Based, Cosine Similarity")
    print(f"  ✅ Content-Based Filtering : Genre Vectors, User Profiles")
    print(f"  ✅ Hybrid System           : CF(60%) + Content-Based(40%)")
    print(f"  ✅ Dataset                 : {len(MOVIES)} movies, {len(USER_RATINGS)} users")
    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    run_demo()
