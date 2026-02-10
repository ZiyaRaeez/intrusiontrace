import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


def apply_scaling(X_train, X_test):
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler



def apply_pca(X_train_scaled, X_test_scaled, n_components=8):
    pca = PCA(
        n_components=n_components,
        random_state=42,
        svd_solver="randomized"
    )

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, pca




def apply_kmeans(X_train_pca, X_test_pca):
    kmeans = MiniBatchKMeans(
        n_clusters=2,
        batch_size=2048,
        random_state=42
    )

    # Fit only on subset (like you did)
    X_subset = X_train_pca[:50000]
    kmeans.fit(X_subset)

    train_clusters = kmeans.predict(X_train_pca)
    test_clusters = kmeans.predict(X_test_pca)

    # Add cluster as extra feature
    X_train_final = np.column_stack((X_train_pca, train_clusters))
    X_test_final = np.column_stack((X_test_pca, test_clusters))

    return X_train_final, X_test_final, kmeans
