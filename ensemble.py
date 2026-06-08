import os
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Bwd Packet Length Max',
    'Bwd Packet Length Mean', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Fwd PSH Flags', 'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Average Packet Size',
]

def _calibrate(benign_sorted, raw):
    return np.searchsorted(benign_sorted, raw, side="right") / len(benign_sorted)

class FlowPrep:
    #Clean and scale. Learns from benign training data and reapplies the exact same transform live.
    def fit(self, X):
        X = self._clean(np.asarray(X, dtype=float))
        self.medians_ = np.nan_to_num(np.nanmedian(X, axis=0))
        
        bad = np.isnan(X)
        if bad.any():
            X[bad] = np.take(self.medians_, np.where(bad)[1])
        X = np.nan_to_num(X)

        self.log_cols_ = X.min(axis=0) >= 0 #log only non-negative columns
        Xt = self._logtf(X)
        self.keep_ = Xt.std(axis=0) > 1e-8  #drop constant-in-benign columns
        self.scaler_ = StandardScaler().fit(Xt[:, self.keep_])
        return self

    def transform(self, X):
        bad = np.isnan(self._clean(np.atleast_2d(np.asarray(X, dtype=float))))
        if bad.any():
            X[bad] = np.take(self.medians_, np.where(bad)[1])
        X = np.nan_to_num(X)
        return self.scaler_.transform(self._logtf(X)[:, self.keep_])

    def _clean(self, X):
        X = X.copy(); X[~np.isfinite(X)] = np.nan; return X

    def _logtf(self, X):
        X = X.copy()
        X[:, self.log_cols_] = np.log1p(np.clip(X[:, self.log_cols_], 0, None))
        return X


class Ensemble:
    def __init__(self, contamination=0.05, pca_components=10, random_state=42):
        self.contamination = contamination
        self.pca_components = pca_components
        self.random_state = random_state
        self.prep = None
        self.ae = None
        self.iso = None
        self.pca = None
        self.threshold = 0.5

    #autoencoder
    def _build_ae(self, input_dim):
        import tensorflow as tf
        ae = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', input_shape=(input_dim,)),  #encoder
            tf.keras.layers.Dense(12, activation='relu'),                            #bottleneck
            tf.keras.layers.Dense(20, activation='relu'),                            #decoder
            tf.keras.layers.Dense(input_dim, activation='linear'),                   #output
        ])
        ae.compile(optimizer='adam', loss='mse')
        return ae

    def _raw_ae(self, Xs):
        recon = self.ae.predict(Xs, verbose=0)
        return np.mean((Xs - recon) ** 2, axis=1)

    def _raw_if(self, Xs):
        return -self.iso.decision_function(Xs)

    def _raw_pca(self, Xs):
        recon = self.pca.inverse_transform(self.pca.transform(Xs))
        return np.mean((Xs - recon) ** 2, axis=1)

    def fit(self, X_benign, epochs=50, batch_size=256, verbose=0):
        #X_benign = raw (unscaled) benign feature rows, shape (n, 25).
        self.prep = FlowPrep().fit(X_benign)
        Xs = self.prep.transform(X_benign)
        n = Xs.shape[1]
        print(f"FlowPrep kept {n}/{len(FEATURES)} features")

        #autoencoder
        import tensorflow as tf
        self.ae = self._build_ae(n)
        self.ae.fit(Xs, Xs, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                    verbose=verbose,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

        #isolation forest
        self.iso = IsolationForest(contamination=self.contamination, n_estimators=200,
                                   random_state=self.random_state, n_jobs=-1).fit(Xs)
        
        #PCA
        self.pca = PCA(n_components=min(self.pca_components, n - 1),
                       random_state=self.random_state).fit(Xs)

        self._cal_ae = np.sort(self._raw_ae(Xs))
        self._cal_if = np.sort(self._raw_if(Xs))
        self._cal_pca = np.sort(self._raw_pca(Xs))

        self.threshold = float(np.quantile(self.score(X_benign), 0.99))
        return self

    def score(self, X):
        #X = raw 25-feature rows (or one row). Returns anomaly score(s) in [0,1].
        Xs = self.prep.transform(X)
        a = _calibrate(self._cal_ae, self._raw_ae(Xs))
        i = _calibrate(self._cal_if, self._raw_if(Xs))
        p = _calibrate(self._cal_pca, self._raw_pca(Xs))
        return (a + i + p) / 3.0

    def predict(self, X, threshold=None):
        t = self.threshold if threshold is None else threshold
        return (self.score(X) >= t).astype(int)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.ae.save(os.path.join(path, "autoencoder.keras"))
        joblib.dump({
            "prep": self.prep, "iso": self.iso, "pca": self.pca,
            "cal_ae": self._cal_ae, "cal_if": self._cal_if, "cal_pca": self._cal_pca,
            "threshold": self.threshold, "features": FEATURES,
            "contamination": self.contamination, "pca_components": self.pca_components,
        }, os.path.join(path, "ensemble.pkl"))
        print(f"Saved ensemble to {path}/ (autoencoder.keras + ensemble.pkl)")
