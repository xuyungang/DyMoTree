import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math
import numpy as np
from pygam import LinearGAM, s


def plot_feature_trend(adata,pseudotime_key,features,color_map,lin_space=200,n_splines=20,lam=50,confidence_level = 0.95,alpha=0.2,linewidth=2.0,figsize=(5, 3),show=True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

    for i, gene_name in enumerate(features):
    
        x_pseudotime = adata.obs[pseudotime_key].values
        #y_expression = adata[:, gene_name].X.toarray().flatten() if hasattr(adata[:, gene_name].X, 'toarray') else adata[:, gene_name].X.flatten()

        if gene_name in adata.var_names:
            X_gene = adata[:, gene_name].X
            y_expression = X_gene.toarray().flatten() if hasattr(X_gene, 'toarray') else X_gene.flatten()
        elif gene_name in adata.obs.columns:
            y_expression = adata.obs[gene_name].values
        else:
            print(f"Warning: '{gene_name}' not found in adata.var_names or adata.obs, skipped.")
            continue

        if len(x_pseudotime) < 15:
            continue

        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(x_pseudotime, y_expression)
        x_smooth = np.linspace(x_pseudotime.min(), x_pseudotime.max(), lin_space)
        y_smooth = gam.predict(x_smooth)
        intervals = gam.confidence_intervals(x_smooth, width=confidence_level)
        
        ax.plot(
            x_smooth, 
            y_smooth, 
            linewidth=linewidth, 
            label=gene_name,  
            color=color_map[gene_name]
        )
        
        ax.fill_between(
            x_smooth, 
            intervals[:, 0], 
            intervals[:, 1], 
            color=color_map[gene_name],
            alpha=alpha 
        )
    ax.set_ylabel('Gene Expression')
    ax.legend()
    if show:
        plt.tight_layout()
        plt.show()
    else:
        return ax