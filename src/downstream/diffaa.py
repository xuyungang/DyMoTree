import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
import archetypes
import time
from datetime import datetime

class _Diffusion():
    def __init__(self, 
                 n_components: int, 
                 k: int, 
                 density_norm: bool = True):
        self.n_components = n_components
        self.k = k
        self.density_norm = density_norm


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        k_for_nn = min(self.k + 1, n_samples)
        nn = NearestNeighbors(n_neighbors=k_for_nn, n_jobs=-1).fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        local_sigmas = distances[:, -1] + 1e-10

        dists_sq = np.sum((X_scaled[:, np.newaxis, :] - X_scaled[np.newaxis, :, :]) ** 2, axis=-1)
        sigma_prods = local_sigmas[:, np.newaxis] * local_sigmas[np.newaxis, :]
        K = np.exp(-dists_sq / sigma_prods)

        if self.density_norm:
            q = K.sum(axis=1)
            q[q == 0] = 1
            D_q_inv = np.diag(1.0 / q)
            T_alpha1 = D_q_inv @ K @ D_q_inv
            
            d_new = T_alpha1.sum(axis=1)
            d_new[d_new == 0] = 1
            D_new_inv = np.diag(1.0 / d_new)
            M = D_new_inv @ T_alpha1
        else: 
            D_diag = K.sum(axis=1)
            D_diag[D_diag == 0] = 1 
            D_inv = np.diag(1.0 / D_diag)
            M = D_inv @ K
            
        M_sparse = csc_matrix(M)
        eigenvalues, eigenvectors = eigs(M_sparse, k=self.n_components + 1, which='LR')
            
        sorted_indices = np.argsort(-np.real(eigenvalues))
        eigenvalues = np.real(eigenvalues[sorted_indices])
        eigenvectors = np.real(eigenvectors[:, sorted_indices])

        embedding = eigenvectors[:, 1:] * eigenvalues[1:]
        return embedding


class Find_State():
    def __init__(self,D,F,n_pca,n_diff,n_gene,n_neighbor,n_state,n_init,max_iter,tol,method='spearman'):
        self.D = D
        self.F = F
        
        self.n_gene = n_gene
        self.n_pca = n_pca
        self.n_neighbor = n_neighbor
        self.n_diff = n_diff
        
        self.n_state = n_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        
        
    def fit(self):
        
        fate_df = self.F
        data_df = self.D
        
        # filtering Data
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Filtering Data by fate space") 
        time.sleep(1) 
                
        all_top_indices = []
        for fate_idx in fate_df.columns:

            correlations = data_df.corrwith(fate_df[fate_idx],method=self.method)
            top_indices_for_fate = correlations.nlargest(self.n_gene).index.tolist()
            all_top_indices.extend(top_indices_for_fate)

        unique_top_indices = sorted(list(set(all_top_indices)))
        F = self.D.values[:, unique_top_indices]
        
        # PCA transformation 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] PCA transformation") 
        time.sleep(1) 
                
        F_scaled = StandardScaler().fit_transform(F)
        pca = PCA(n_components=self.n_pca)
        P = pca.fit_transform(F_scaled)
        
        # Diffusion map transformation
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Run Diffusion map") 
        time.sleep(1) 
                
        destiny_analyzer = _Diffusion(n_components=self.n_diff, k=self.n_neighbor)
        diffusion_embedding = destiny_analyzer.fit_transform(P)
        self.diff = diffusion_embedding

        # Find archetypes
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Find archetypes") 
        time.sleep(1) 
                
        aa = archetypes.AA(
            n_archetypes=self.n_state,
            n_init=self.n_init,         # for robust result
            max_iter=self.max_iter,
            tol=self.tol
        )
        aa.fit(diffusion_embedding)
        self.aa = aa
        
    def get_diff(self,axis = [0,1]):
        
        return self.diff[:, axis]

    def get_state(self):
        
        atypes = self.aa.transform(X=self.diff)
        
        return  np.argmax(atypes, axis=1)
