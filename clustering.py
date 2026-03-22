"""
Player Archetype Clustering for V2 model.

Clusters players on style/deployment features (NOT the target) using
rolling windows. Creates player archetypes like "slot shooter",
"perimeter sniper", "PP specialist", "grinder", etc.

Clustering is refitted at training time; at prediction time, players
are assigned to the nearest existing centroid.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).resolve().parent / "saved_model_v2"

# Features used for clustering — style/deployment only, never the target
CLUSTER_FEATURES = [
    "avg_toi",
    "pp_toi_pct",
    "slot_pct",
    "avg_shot_distance",
    "rebound_rate",
    "rush_rate",
    "shots_per_60",
    "avg_shift_length",
]

# Range of k values to test
K_RANGE = range(4, 9)  # 4,5,6,7,8


class PlayerClusterer:
    """Fits and applies player archetype clusters."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.k = None
        self.cluster_profiles = {}  # cluster_id -> {feature means + sog mean}
        self.feature_names = CLUSTER_FEATURES

    def fit(self, player_features: np.ndarray, player_sog: np.ndarray = None,
            k: int = None) -> int:
        """Fit clustering model on player feature matrix.

        Args:
            player_features: (n_players, n_features) array of style features.
            player_sog: Optional SOG values per player (for cluster profiles, NOT for clustering).
            k: Force specific k. If None, selects best by silhouette score.

        Returns:
            Chosen k value.
        """
        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(player_features)

        if k is not None:
            self.k = k
            self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
            self.model.fit(X)
        else:
            # Select k by silhouette score
            best_k = 5
            best_score = -1

            for k_test in K_RANGE:
                if k_test >= len(X):
                    continue
                km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
                labels = km.fit_predict(X)

                # Need at least 2 non-trivial clusters
                if len(set(labels)) < 2:
                    continue

                score = silhouette_score(X, labels, sample_size=min(5000, len(X)))
                logger.info("  k=%d: silhouette=%.4f", k_test, score)

                if score > best_score:
                    best_score = score
                    best_k = k_test

            self.k = best_k
            self.model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.model.fit(X)
            logger.info("Selected k=%d (silhouette=%.4f)", best_k, best_score)

        # Build cluster profiles
        labels = self.model.labels_
        self.cluster_profiles = {}
        for cid in range(self.k):
            mask = labels == cid
            profile = {
                "n_players": int(mask.sum()),
                "feature_means": player_features[mask].mean(axis=0).tolist(),
            }
            if player_sog is not None:
                profile["mean_sog"] = float(player_sog[mask].mean())
                profile["std_sog"] = float(player_sog[mask].std())
            self.cluster_profiles[cid] = profile

        return self.k

    def predict(self, player_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assign players to clusters and compute distances.

        Args:
            player_features: (n_players, n_features) or (n_features,) array.

        Returns:
            (cluster_ids, distances_to_centroid)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Clusterer not fitted")

        if player_features.ndim == 1:
            player_features = player_features.reshape(1, -1)

        X = self.scaler.transform(player_features)
        cluster_ids = self.model.predict(X)
        distances = self.model.transform(X)  # (n, k) distances to each centroid
        min_distances = np.array([distances[i, c] for i, c in enumerate(cluster_ids)])

        return cluster_ids, min_distances

    def get_cluster_mean_sog(self, cluster_id: int) -> float:
        """Get mean SOG for a cluster."""
        profile = self.cluster_profiles.get(cluster_id)
        if profile and "mean_sog" in profile:
            return profile["mean_sog"]
        return 2.0  # safe default

    def save(self, path: Path = None):
        """Save cluster model to disk."""
        if path is None:
            path = SAVE_DIR
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "cluster_model.pkl", "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "k": self.k,
                "cluster_profiles": self.cluster_profiles,
                "feature_names": self.feature_names,
            }, f)
        logger.info("Cluster model saved (k=%d)", self.k)

    def load(self, path: Path = None) -> bool:
        """Load cluster model from disk. Returns True if successful."""
        if path is None:
            path = SAVE_DIR
        pkl_path = path / "cluster_model.pkl"

        if not pkl_path.exists():
            return False

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.k = data["k"]
        self.cluster_profiles = data["cluster_profiles"]
        self.feature_names = data.get("feature_names", CLUSTER_FEATURES)
        logger.info("Cluster model loaded (k=%d)", self.k)
        return True

    def describe_clusters(self) -> str:
        """Generate a human-readable description of clusters."""
        if not self.cluster_profiles:
            return "No clusters fitted."

        lines = [f"Player Archetypes (k={self.k})", ""]
        for cid in sorted(self.cluster_profiles):
            p = self.cluster_profiles[cid]
            means = p["feature_means"]
            sog = p.get("mean_sog", "?")
            lines.append(f"Cluster {cid}: {p['n_players']} players, avg SOG={sog:.2f}" if isinstance(sog, float) else f"Cluster {cid}: {p['n_players']} players")

            for i, feat in enumerate(self.feature_names):
                if i < len(means):
                    lines.append(f"  {feat}: {means[i]:.3f}")
            lines.append("")

        return "\n".join(lines)
