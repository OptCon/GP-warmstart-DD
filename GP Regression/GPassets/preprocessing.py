from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle 
import numpy as np


class PreProcessor:
    """
    Python class for data pre-processing
    """
    def __init__(self):
        self.x_scaler = None
        self.y_scaler = None
        self.pca = None
        self.var_thresh = 0.99
        self.rand_seed = 0
    # ---------------------------- Internal functions ----------------------------------------------------
    def _as_2d(self, arr, *, name: str, expected_features: int | None = None):
        """
        Method to ensure numpy array is 2D in the shape (N, d).
        Accepts:
          - (d,)           -> (1, d)
          - (d, 1)         -> (1, d)  (only if expected_features == d)
          - (N, d)         -> unchanged
        Rejects ambiguous / wrong shapes with ValueError.
        """
        a = np.asarray(arr)

        if a.ndim == 1:
            return a.reshape(1, -1)

        if a.ndim != 2:
            raise ValueError(f"{name} must be a 1D or 2D array; got shape {a.shape} (ndim={a.ndim}).")

        # Handle column vector (d,1) for single sample if we know d
        if expected_features is not None and a.shape[1] == 1 and a.shape[0] == expected_features:
            return a.reshape(1, expected_features)

        return a

    def _require_fitted(self, obj, obj_name: str, hint: str):
        """
        Method to ensure object existence before fitting
        """
        if obj is None:
            raise ValueError(f"{obj_name} has not been created. Call {hint} first (or load a saved prep assets file).")
    
    # ---------------------------------------- Public functions -------------------------------------------------------------
    def prep_data(self, X, Y, *, k_cap : int | None = None, debug : bool | None = None):
        """
        Docstring for prep_data
        
        :param X: Raw input data
        :param Y: Raw output data
        :param k_cap: Maximum  number of allowed models (Debugging tool)
        :param debug: Debugging flag
        """
        X = self._as_2d(X, name='X')
        Y = self._as_2d(Y, name='Y') 

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same number of samples. Got X: {X.shape}, Y: {Y.shape}.")

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, shuffle=True, random_state=self.rand_seed)

        # Create x_scaler and y_scaler objects
        self.x_scaler = StandardScaler().fit(x_train)
        self.y_scaler = StandardScaler().fit(y_train)

        # Standardize x_train
        x_train_scaled = self.scale_X(x_raw = x_train)

        # Standardize x_test
        x_test_scaled = self.scale_X(x_raw = x_test)

        # Standardize y_train
        y_train_scaled = self.scale_Y(y_raw = y_train)

        # Create PCA object
        if k_cap is None:
            self.pca = PCA(n_components=self.var_thresh).fit(y_train_scaled)
        else:
            self.pca = PCA(n_components=k_cap).fit(y_train_scaled)
        # Apply PCA on y_train_scaled
        y_train_pca = self.pca_Y(y_train_scaled)

        # Debugging
        if debug:
            print(f"Reduced {y_train.shape[-1]} duals to {self.pca.n_components_} latent variables.")
            print(f"Variance explained by components: {self.pca.explained_variance_ratio_}")

            total_explained = np.sum(self.pca.explained_variance_ratio_)
            print(f"Total variance captured by {self.pca.n_components_} components: {total_explained:.4f}")

            pca_full = PCA().fit(y_train_scaled)
            if self.pca.n_components_ != y_train.shape[1]:
                print(f"Variance of {self.pca.n_components_+1} component: {pca_full.explained_variance_ratio_[self.pca.n_components_]:.6f}")

            Z = self.pca_Y(y_train_scaled)
            y_recon_scaled = self.reverse_pca_Y(Z)
            y_recon_full = self.reverse_scale_Y(y_recon_scaled)

            max_error = np.max(np.abs(y_train - y_recon_full))
            print(f"Max reconstruction error in duals: {max_error}")

        return x_train_scaled, x_test_scaled, y_train_pca, y_test

    def scale_X(self, x_raw):
        """
        Method to scale x 
        :param x_raw: Raw input data
        """
        self._require_fitted(self.x_scaler, 'x_scaler', "prep_data(...)")
        
        # expected feature count = d
        d = int(self.x_scaler.mean_.shape[0])
        x = self._as_2d(x_raw, name="x_raw", expected_features=d)

        # Reject wrong feature count early
        if x.shape[1] != d:
            raise ValueError(f"x_raw must have {d} features (shape (N,{d})); got shape {x.shape}.")

        return self.x_scaler.transform(x)
    
    def scale_Y(self, y_raw):
        """
        Method to scale y
        
        :param y_raw: Raw output data
        """
        self._require_fitted(self.y_scaler, 'y_scaler', 'prep_data(...)')

        # Check dimensions
        m = int(self.y_scaler.mean_.shape[0])
        y = self._as_2d(y_raw, name="y_raw", expected_features=m)

        if y.shape[1] != m:
            raise ValueError(f"y_raw must have {m} features (shape (N,{m})); got shape {y.shape}.")


        return self.y_scaler.transform(y) 

    def pca_Y(self, y_scaled):
        """
        Method to perform PCA on the scaled output data

        :param y_scaled: Scaled output data
        """
        self._require_fitted(self.pca, "pca", "prep_data(...)")

        y = self._as_2d(y_scaled, name="y_scaled")

        # PCA expects the same number of features it was fit on
        n_in = int(self.pca.n_features_in_)
        if y.shape[-1] != n_in:
            raise ValueError(
                f"y_scaled must have {n_in} features to match PCA fit; got shape {y.shape}."
            )

        return self.pca.transform(y)
        
    def reverse_scale_Y(self, y_scaled):
        """
        Method to reverse scaling of y
        :param y_scaled: Scaled output data
        """
        self._require_fitted(self.y_scaler, "y_scaler", "prep_data(...)")

        y = self._as_2d(y_scaled, name="y_scaled")
        m = int(self.y_scaler.mean_.shape[0])

        if y.shape[-1] != m:
            raise ValueError(f"y_scaled must have {m} features (shape (N,{m})); got shape {y.shape}.")

        return self.y_scaler.inverse_transform(y)

    def reverse_pca_Y(self, y_pca):
        """
        Method to reverse the PCA process on the latent output data
        
        :param y_pca: Data in latent space
        """
        self._require_fitted(self.pca, "pca", "prep_data(...)")

        y = self._as_2d(y_pca, name="y_pca")
        m_latent = int(self.pca.n_components_)

        if y.shape[-1] != m_latent:
            raise ValueError(f"y_pca must have {m_latent} features (shape (N,{m_latent})); got shape {y.shape}.")

        return self.pca.inverse_transform(y)

    def save(self, savepath):
        """
        Method to save prep assets for validation
        
        :param savepath: Save path
        """
        self._require_fitted(self.x_scaler, "x_scaler", "prep_data(...)")
        self._require_fitted(self.y_scaler, "y_scaler", "prep_data(...)")
        self._require_fitted(self.pca, "pca", "prep_data(...)")

        # Save preprocessing assets
        with open(savepath, 'wb') as f:
            pickle.dump({
                'x_scaler' : self.x_scaler,
                'y_scaler' : self.y_scaler,
                'pca' : self.pca
            }, f) 

    def load(self, savepath):
        """
        Method to load prep assets for validation
        
        :param savepath: Save path
        """
        # Load Preprocessing assets
        with open(savepath,'rb') as f:
            assets = pickle.load(f)

        for k in ("x_scaler", "y_scaler", "pca"):
            if k not in assets:
                raise ValueError(f"Invalid preprocessing file: missing key '{k}'.")
            
        self.x_scaler = assets['x_scaler']
        self.y_scaler = assets['y_scaler']
        self.pca = assets['pca']