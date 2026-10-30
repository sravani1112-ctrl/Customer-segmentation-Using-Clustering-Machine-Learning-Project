from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def get_wcss(X, max_k=10):

    k_values = []
    wcss_values = []

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        k_values.append(k)
        wcss_values.append(model.inertia_)

    return k_values, wcss_values


def perform_kmeans(X, k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels


def get_silhouette_score(X, labels):

    unique_labels = set(labels)

    if len(unique_labels) < 2:
        return 0

    return silhouette_score(
        X,
        labels
    )


def perform_dbscan(
    X,
    eps=0.5,
    min_samples=5
):

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)

    return model, labels