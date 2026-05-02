"""Feature extractors: PCA, HOG, color histograms."""
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca(X_train, X_val, X_test, n_components):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)
    pca = PCA(n_components=n_components, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_val_p = pca.transform(X_val_s)
    X_te_p = pca.transform(X_te_s)
    return X_tr_p, X_val_p, X_te_p, pca, scaler


def extract_hog_single(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(gray).flatten()


def extract_hog_features(images):
    return np.array([extract_hog_single(images[i]) for i in range(len(images))])


def extract_color_histograms(images, bins=32):
    hists = []
    for i in range(len(images)):
        img = images[i]
        ch_hists = []
        for ch in range(3):
            h, _ = np.histogram(img[:, :, ch], bins=bins, range=(0.0, 1.0))
            h = h.astype(np.float32)
            h /= h.sum() + 1e-7
            ch_hists.append(h)
        hists.append(np.concatenate(ch_hists))
    return np.array(hists)


def extract_handcrafted_features(images, hog=True, color_hist=True, bins=32):
    parts = []
    if hog:
        parts.append(extract_hog_features(images))
    if color_hist:
        parts.append(extract_color_histograms(images, bins))
    return np.concatenate(parts, axis=1)
