# AudienceDecode

A cluster-based collaborative filtering recommendation system for large-scale streaming data, built on top of ~100 million anonymized user-movie interactions.

---

## The Problem

Large streaming platforms generate enormous volumes of interaction data that are rarely exploited beyond simple popularity rankings. Raw ratings are noisy, biased by user-level tendencies, and inconsistent for movies with few reviews. The challenge is to extract stable, interpretable preference patterns from this data and translate them into personalized recommendations, without relying on content metadata or individual interaction history.

This project addresses that challenge by learning behavioral archetypes from aggregated statistics, then routing recommendations through those archetypes rather than individual user-item pairs.

---

## What We Built

AudienceDecode is a dual-clustering recommendation pipeline that:

1. **Segments 434,429 users into 4 behavioral clusters** based on engagement volume, rating consistency, and activity duration.
2. **Segments 10,000 movies into 3 content clusters** based on popularity, Bayesian-adjusted quality, rating dispersion, and polarization.
3. **Constructs a 4x3 preference matrix** where each cell measures the bias-corrected affinity between a user segment and a content category, weighted by interaction volume.
4. **Scores and ranks all unseen movies** for any user via a linear combination of cluster affinity (alpha = 0.6) and individual movie quality (gamma = 0.4), returning a Top-N ranked list.

The system is fully interpretable: every recommendation can be traced back to observable cluster centroids and explicit affinity weights, with no latent factors involved.

---

## Key Results

### Clustering

The optimal configuration, selected by consensus across Silhouette analysis and the Elbow method, is **k = 4 for users** and **k = 3 for movies**.

**Algorithm comparison, movies (k = 3):**

| Model             | Centroid Silhouette | Min Cluster Size | Std Cluster Size | Time (s) | Composite Score |
|-------------------|---------------------|------------------|------------------|----------|-----------------|
| MiniBatch K-Means | 0.536               | 3,689            | 1,375            | 0.162    | 1.000           |
| K-Means           | 0.524               | 19               | 4,254            | 0.169    | -0.333          |
| BIRCH             | 0.814               | 14               | 7,493            | 0.178    | -0.667          |

**Algorithm comparison, users (k = 4):**

| Model             | Centroid Silhouette | Min Cluster Size | Std Cluster Size | Time (s) | Composite Score |
|-------------------|---------------------|------------------|------------------|----------|-----------------|
| MiniBatch K-Means | 0.437               | 54,538           | 48,732           | 5.252    | 0.000           |
| K-Means           | 0.434               | 54,988           | 53,069           | 4.919    | 0.000           |
| BIRCH             | 0.481               | 655              | 164,499          | 4.612    | 0.000           |

**MiniBatch K-Means** is selected as the best algorithm for both clustering tasks. For movies, it achieves the highest composite score (1.000), delivering well-balanced clusters (minimum size 3,689) and clean separation (Silhouette 0.536). For users, it ties K-Means on composite score but edges ahead on Silhouette (0.437 vs 0.434) and cluster balance (std 48,732 vs 53,069). BIRCH achieves the highest raw Silhouette in both cases but produces severely imbalanced clusters (minimum sizes of 14 for movies and 655 for users), which undermines interpretability.

### Recommendation Evaluation

**Score calibration** (predicted preference score vs actual normalized rating, evaluated on 45,641 interactions):

| Metric              | Value  |
|---------------------|--------|
| RMSE                | 0.2806 |
| MAE                 | 0.2315 |
| Pearson correlation | 0.2494 |

**Ranking quality** (temporal hold-out, last 20% of ratings per user as test set, relevance threshold rating >= 4.0, evaluated on 427 users):

| k  | Precision@k | Recall@k | Hit Rate@k |
|----|-------------|----------|------------|
| 5  | 0.0623      | 0.1173   | 0.2740     |
| 10 | 0.0546      | 0.2106   | 0.4286     |
| 20 | 0.0422      | 0.3329   | 0.6066     |

Hit Rate@20 of 0.61 means that 61% of users would receive at least one genuinely liked movie (rated 4 or above on a 0-6 scale) in a list of 20 recommendations, which is the practically relevant target for a cluster-level system. Because the model operates entirely at the group level, some irreducible per-user error is expected. A user-level model such as matrix factorization or neural collaborative filtering, that captures within-cluster individual variation, would be expected to score higher on all three metrics.

---

## Methodology

### Dataset

`viewer_interactions.db` is a SQLite database with four tables:

- **`viewer_ratings`**: ~100 million interaction records (user ID, movie ID, rating 0-6, timestamp)
- **`movies`**: 10,000 movies with title and release year
- **`user_statistics`**: pre-aggregated user behavioral metrics
- **`movie_statistics`**: pre-aggregated movie reception metrics

### Feature Engineering

**User features** (5 dimensions after engineering):

| Feature              | Description                                          |
|----------------------|------------------------------------------------------|
| `total_ratings`      | Interaction volume                                   |
| `avg_rating`         | Mean rating given                                    |
| `std_rating`         | Rating variability                                   |
| `rating_consistency` | `avg_rating / (std_rating + epsilon)`: stable raters score higher |
| `activity_days`      | Time span between first and last rating              |

`unique_movies` is not used directly: its information is captured by `total_ratings` (correlation > 0.9) and adding it would double-weight engagement volume in the distance metric without adding signal.

**Movie features** (4 dimensions after engineering):

| Feature               | Description                                                       |
|-----------------------|-------------------------------------------------------------------|
| `total_ratings`       | Interaction volume                                                |
| `bayesian_avg_rating` | Shrinks raw average toward the global mean for low-count movies   |
| `std_rating`          | Rating dispersion                                                 |
| `rating_range`        | `max_rating - min_rating`: captures audience polarization         |

Raw `avg_rating` is replaced by `bayesian_avg_rating` using the formula `(v * R + C * m) / (v + C)`, where `v` is the movie's total rating count, `R` is its raw average, `C` is the median count across all movies, and `m` is the global average. This prevents single 5-star ratings from outranking movies with thousands of reviews.

### Preprocessing

1. Remove records with missing keys, invalid ratings (outside 0-6), and duplicate entries.
2. Filter anomalous timestamps; impute missing `year_of_release` with the median.
3. Reconstruct `activity_days` from `first_rating_days` and `last_rating_days`.
4. Cap features at the 99th percentile to reduce skew without discarding data.
5. Median imputation for remaining missing values.
6. `StandardScaler` normalization before clustering.

### Clustering and Model Selection

Three centroid-based algorithms are evaluated across k in [3, 9]:

- **K-Means**: standard Lloyd's algorithm (`n_init=10`, `max_iter=300`)
- **MiniBatch K-Means**: mini-batch variant for scalability (`batch_size=10000`, `n_init=5`)
- **BIRCH**: tree-based incremental clustering (`threshold=0.5`, `branching_factor=50`)

Optimal k is chosen by majority vote across algorithms on centroid Silhouette peaks, confirmed visually with Elbow curves. The Silhouette score is computed with a vectorized O(n log k) approximation using centroid distances instead of the standard O(n^2) pairwise approach.

The best algorithm is then selected using a composite score that normalizes and combines Silhouette, minimum cluster size, cluster size standard deviation, and computation time across all three candidates. The selection is fully dynamic: the pipeline reads `best_user_model_name` and `best_movie_model_name` from the comparison table at runtime rather than hardcoding a choice.

### Preference Matrix and Scoring

Cluster-level affinity between user cluster `u` and movie cluster `m` is computed as:

```
excess_rating(u, m) = mean_rating(u, m) - mean_rating(u)
cluster_weight(u, m) = (excess_rating(u, m) - min_excess) * log(1 + interaction_count(u, m))
```

Subtracting the per-user-cluster baseline removes rating-level bias: user segments that systematically give high ratings would otherwise appear to prefer every movie cluster equally. The result is weighted by log interaction volume so that high-confidence pairs dominate the matrix.

Each movie receives a quality score:

```
quality_score = 0.7 * normalized(bayesian_avg_rating) + 0.3 * normalized(log(total_ratings))
```

The final recommendation score for a (user, movie) pair is:

```
score = alpha * cluster_weight(u_cluster, m_cluster) + gamma * quality_score(movie)
alpha = 0.6,  gamma = 0.4
```

Already-seen movies are excluded, and the top-N candidates by score are returned.

### Cluster Profiles

**User segments (k = 4, MiniBatch K-Means):**

- **U0, Selective Mainstream Explorers**: Strong preference for F1 (0.50) and F2 (0.44), near-zero for F0 (0.05).
- **U1, High-Engagement Quality Seekers**: Highest overall preference intensity, F2 (0.84) and F1 (0.77) dominant.
- **U2, Casual Quality-Oriented Viewers**: Similar ranking to U1 but lower intensity, F2 (0.79) > F1 (0.59) >> F0 (0.08).
- **U3, Highly Selective Critics**: Most restrictive, F0 scores 0.00, F2 (0.72) and F1 (0.56) preferred.

**Movie categories (k = 3, MiniBatch K-Means):**

- **F0, Low-Attraction Marginal Content**: Low ratings, low interaction volume, limited audience reach.
- **F1, Mainstream Popular Titles**: Broad reach, moderate-to-high ratings, commercially successful.
- **F2, Top-Quality High-Engagement Movies**: High Bayesian-adjusted rating combined with large interaction volume; platform flagship content.

Across all user segments, F0 is systematically avoided and F2 is consistently preferred, confirming that the preference matrix captures a real and stable signal rather than noise.

---

## Stack and Libraries

| Library        | Version | Purpose                                        |
|----------------|---------|------------------------------------------------|
| `numpy`        | 2.0.2   | Vectorized computation, array operations       |
| `pandas`       | 2.2.2   | DataFrame manipulation, groupby aggregations   |
| `scikit-learn` | 1.6.1   | K-Means, MiniBatch K-Means, BIRCH, StandardScaler, PCA |
| `matplotlib`   | 3.10.0  | Elbow curves, PCA projections, heatmaps        |
| `seaborn`      | 0.13.2  | Correlation heatmaps, distribution plots       |
| `sqlite3`      | stdlib  | Database connection (no install required)      |

Python 3.x. Full version pins are in `requirements.txt`.

---

## Project Structure

```
AudienceDecode/
├── AudienceDecode.ipynb       # Main notebook: full pipeline with outputs
├── viewer_interactions.db     # SQLite database (not tracked in git)
├── requirements.txt           # Pinned Python dependencies
├── README.md                  # This file
└── images/
    ├── user_engagement_dist.png
    ├── correlation_heatmap_users.png
    ├── elbow_silhouette_users_kmeans.png
    ├── elbow_silhouette_users_minibatch.png
    ├── elbow_silhouette_users_birch.png
    ├── elbow_silhouette_movies_kmeans.png
    ├── elbow_silhouette_movies_minibatch.png
    ├── elbow_silhouette_movies_birch.png
    ├── pca_projection_users.png
    ├── pca_projection_movies.png
    └── preference_matrix.png
```

---

## Setup

**1. Clone the repository and install dependencies:**

```bash
git clone <repo-url>
cd AudienceDecode
pip install -r requirements.txt
```

**2. Place the database file in the project root:**

The notebook expects `viewer_interactions.db` in the same directory as `AudienceDecode.ipynb`. The database is not tracked in version control due to its size.

**3. Run the notebook:**

```bash
jupyter notebook AudienceDecode.ipynb
```

Or execute it headlessly and write outputs back in place:

```bash
jupyter nbconvert --to notebook --execute --inplace AudienceDecode.ipynb
```

**4. Generate recommendations for a specific user:**

In Section 7 of the notebook, set `target_user_id` to any valid user ID (there are 434,429 in the dataset) and re-run the cell. The output table ranks the top-N unseen movies by score.

---

## Academic Context

Developed as part of the Machine Learning course at LUISS Guido Carli University, academic year 2025/2026.

**Team:**
- Arianna Cambi
- Sofia Capriolo
- Andrea Cipolla
- Giorgio Vanini
