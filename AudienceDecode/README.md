# AudienceDecode

A cluster-based collaborative filtering recommendation system for large-scale streaming data, built on top of ~100 million anonymized user-movie interactions.

---

## The Problem

Large streaming platforms generate enormous volumes of interaction data that are rarely exploited beyond simple popularity rankings. Raw ratings are noisy, biased by user-level tendencies, and inconsistent for movies with few reviews. The challenge is to extract stable, interpretable preference patterns from this data and translate them into personalized recommendations, without relying on content metadata or individual interaction history.

This project addresses that challenge by learning behavioral archetypes from aggregated statistics, then routing recommendations through those archetypes rather than individual user-item pairs.

---

## What We Built

AudienceDecode is a dual-clustering recommendation pipeline that:

1. **Segments 434,429 users into 6 behavioral clusters** based on engagement volume, rating consistency, and activity duration.
2. **Segments 10,000 movies into 3 content clusters** based on popularity, Bayesian-adjusted quality, rating dispersion, and polarization.
3. **Constructs a 6x3 preference matrix** where each cell measures the bias-corrected affinity between a user segment and a content category, weighted by interaction volume.
4. **Scores and ranks all unseen movies** for any user via a linear combination of cluster affinity (alpha = 0.6) and individual movie quality (gamma = 0.4), returning a Top-N ranked list.

The system is fully interpretable: every recommendation can be traced back to observable cluster centroids and explicit affinity weights, with no latent factors involved.

---

## Key Results

### Clustering

The optimal configuration, selected by consensus across Silhouette analysis and the Elbow method, is **k = 6 for users** and **k = 3 for movies**.

**Algorithm comparison, users (k = 6):**

| Model             | Centroid Silhouette | Min Cluster Size | Std Cluster Size | Time (s) | Composite Score |
|-------------------|---------------------|------------------|------------------|----------|-----------------|
| MiniBatch K-Means | 0.490               | 25,958           | 53,965           | 0.165    | 0.667           |
| BIRCH             | 0.525               | 2,424            | 137,901          | 0.149    | 0.000           |
| K-Means           | 0.489               | 25,957           | 54,056           | 0.166    | -0.667          |

**Algorithm comparison, movies (k = 3):**

| Model             | Centroid Silhouette | Min Cluster Size | Std Cluster Size | Time (s) | Composite Score |
|-------------------|---------------------|------------------|------------------|----------|-----------------|
| MiniBatch K-Means | 0.505               | 3,738            | 1,390            | 0.006    | 0.333           |
| BIRCH             | 0.787               | 15               | 7,494            | 0.005    | 0.000           |
| K-Means           | 0.529               | 20               | 4,692            | 0.008    | -0.333          |

**MiniBatch K-Means** is selected as the best algorithm for both clustering tasks. For users, it leads on composite score (0.667) with a well-balanced minimum cluster size of 25,958 and the best Silhouette among the practical candidates (0.490). For movies, it again ranks first (0.333), producing the most evenly distributed clusters (minimum size 3,738, std 1,390). BIRCH achieves the highest raw Silhouette in both cases but is penalized for severely imbalanced clusters (minimum sizes of 15 for movies and 2,424 for users), which undermines interpretability.

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

### Exploratory Data Analysis

Distribution of user engagement metrics, showing the heavy-tailed behavior typical of streaming platforms: most users have low activity and a small fraction drives the bulk of interactions.

![User Engagement Distributions](images/user_engagement_dist.png)

Correlation heatmap of user features. `total_ratings` and `unique_movies` correlate at 0.94, confirming the redundancy that motivates removing `unique_movies` from the feature set.

![User Features Correlation](images/correlation_heatmap_users.png)

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

**User clustering, Elbow and Silhouette curves (k in [3, 9]):**

K-Means:

![Elbow and Silhouette - Users K-Means](images/elbow_silhouette_users_kmeans.png)

MiniBatch K-Means:

![Elbow and Silhouette - Users MiniBatch K-Means](images/elbow_silhouette_users_minibatch.png)

BIRCH:

![Elbow and Silhouette - Users BIRCH](images/elbow_silhouette_users_birch.png)

**Movie clustering, Elbow and Silhouette curves (k in [3, 9]):**

K-Means:

![Elbow and Silhouette - Movies K-Means](images/elbow_silhouette_movies_kmeans.png)

MiniBatch K-Means:

![Elbow and Silhouette - Movies MiniBatch K-Means](images/elbow_silhouette_movies_minibatch.png)

BIRCH:

![Elbow and Silhouette - Movies BIRCH](images/elbow_silhouette_movies_birch.png)

**2D PCA projections** of the final cluster assignments (MiniBatch K-Means). Clusters show partial overlap in the reduced space, which is expected given the high-dimensional original feature space, but maintain clear centroid separation.

Users (k = 6):

![PCA Projection - Users](images/pca_projection_users.png)

Movies (k = 3):

![PCA Projection - Movies](images/pca_projection_movies.png)

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

**User segments (k = 6, MiniBatch K-Means):**

- **U0, Mainstream-First with marginal tolerance**: Strong F1 preference (0.97), moderate F2 (0.65), highest F0 tolerance among mid-range clusters (0.16).
- **U1, Moderate Mainstream Viewers**: Lowest overall preference intensity, F1 (0.84) and F2 (0.59), with near-zero F0 (0.03).
- **U2, Mainstream-Focused, Quality-Aware**: Very high F1 (0.98), moderate F2 (0.62), avoids F0 (0.05).
- **U3, Peak Mainstream Engagement**: Highest scores across all categories, F1 at maximum (1.00), F2 (0.66), avoids F0 (0.03).
- **U4, Broad Content Explorers**: High F1 (0.91), moderate F2 (0.59), highest F0 tolerance of all segments (0.21).
- **U5, Selective Mainstream Viewers**: High F1 (0.94), moderate F2 (0.57), zero tolerance for F0 (0.00).

**Movie categories (k = 3, MiniBatch K-Means):**

- **F0, Low-Attraction Marginal Content**: Low ratings, low interaction volume, limited audience reach.
- **F1, Mainstream Popular Titles**: Broad reach, moderate-to-high ratings, commercially successful.
- **F2, Top-Quality High-Engagement Movies**: High Bayesian-adjusted rating combined with large interaction volume; platform flagship content.

Across all user segments, F0 is systematically avoided and F1 (Mainstream Popular Titles) is the most consistently preferred category, with F2 as a secondary preference in every segment. User clusters differentiate primarily in overall preference intensity and in their degree of tolerance for F0.

**User-cluster x Movie-cluster preference matrix.** Cell values are preference scores combining bias-corrected cluster affinity (alpha = 0.6) and Bayesian quality score (gamma = 0.4). The block structure validates that the segmentation reflects genuine preference patterns rather than random assignment.

![Preference Matrix](images/preference_matrix.png)

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
