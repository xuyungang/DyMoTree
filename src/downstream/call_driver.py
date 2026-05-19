import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression,Lasso

class Call_Driver():
    def __init__(self,
                 D: pd.DataFrame,
                 F: pd.DataFrame,
                 soft_treshold = 5,
                 method = 'spearman',
                 lasso_alpha = 0.1,
                 graph_threshold = 0.1,
                 model='linear',
                 top_n = None):

        self.model=model
        self.beta = soft_treshold
        self.g_threshold = graph_threshold
        self.method = method
        self.top_n = top_n
        self.lasso_alpha = lasso_alpha

        self.D = D
        self.F = F
        if self.top_n:
            self.D,self.top_genes = self._filter()


    def fit(self):
        
        def regression(x,y):
            if self.model == 'linear':
                model = LinearRegression()
                model.fit(x, y)
                coef = model.coef_[0][0]
                intercept = model.intercept_[0]
                score = model.score(x, y)
                return coef,intercept,score
            elif self.model == 'lasso':
                model = Lasso(alpha=self.lasso_alpha)
                model.fit(x, y)
                coef = model.coef_#[0][0]
                intercept = model.intercept_#[0]
                score = model.score(x, y)
                return coef,intercept,score

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}]Do regression") 
        time.sleep(1) 

        coef_df = pd.DataFrame(index=self.D.columns,
                       columns=self.F.columns)
        inter_df = pd.DataFrame(index=self.D.columns,
                       columns=self.F.columns)
        score_df = pd.DataFrame(index=self.D.columns,
                       columns=self.F.columns)
        if self.model == 'linear':
            for gene in self.D.columns:
                for fate in self.F.columns:
                    X = self.D[gene].values.reshape((-1,1))
                    Y = self.F[fate].values.reshape((-1,1))
                    coef_,inter_,score_ = regression(X,Y)
                    coef_df.loc[gene,fate] = coef_
                    inter_df.loc[gene,fate] = inter_
                    score_df.loc[gene,fate] = score_
            # convolution by co-expression network
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] convolution by co-expression network") 
            time.sleep(1)
            adj = self._coexpression(threshold=self.g_threshold,beta=self.beta,method=self.method)
            degree_ = adj.values.sum(axis=1,keepdims=True)
            with np.errstate(divide='ignore'):
                inv_degree_ = np.power(degree_, -1)
            inv_degree_[np.isinf(inv_degree_)] = 0
            adj_norm = adj.values * inv_degree_
            #_degree = adj.values.sum(axis=0,keepdims=True)
            
            
            coef_new = np.matmul(coef_df.T,adj_norm)#/_degree
            coef_new.columns = self.D.columns
            inter_new = np.matmul(inter_df.T,adj_norm)#/_degree
            inter_new.columns = self.D.columns
            score_new = np.matmul(score_df.T,adj_norm)#/_degree
            score_new.columns = self.D.columns
            
            a = [self.D.corrwith(self.F[fate], method=self.method).values.reshape((1,-1)) for fate in self.F.columns]
            a = np.concatenate(a,axis=0)
            a[np.isnan(a)] = 0
            a = a.T
            
            self.coef = coef_new.T# * (1+a)
            self.intercept = inter_new.T# * (1+a)
            self.score = score_new.T# * (1+a)
                
        elif self.model == 'lasso':
            for fate in self.F.columns:
                    X = self.D
                    Y = self.F[fate]
                    coef_,_,_ = regression(X,Y)
                    coef_df[fate] = coef_
            # convolution by co-expression network
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] convolution by co-expression network") 
            time.sleep(1)
            adj = self._coexpression(threshold=self.g_threshold,beta=self.beta,method=self.method)
            degree_ = adj.values.sum(axis=1,keepdims=True)
            with np.errstate(divide='ignore'):
                inv_degree_ = np.power(degree_, -1)
            inv_degree_[np.isinf(inv_degree_)] = 0
            adj_norm = adj.values * inv_degree_
            #_degree = adj.values.sum(axis=0,keepdims=True)
            
            
            coef_new = np.matmul(coef_df.T,adj_norm)#/_degree
            coef_new.columns = self.D.columns            
            self.coef = coef_new.T# * (1+a)

     
        

    

    

    
    def _coexpression(self,threshold=0.1,beta=1,method='spearman'):

        correlation_matrix = self.D.corr(method=method)

        #correlation_matrix[correlation_matrix<0] = 0
        adjacency_matrix = correlation_matrix.fillna(0)

        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[0]):
                
                if adjacency_matrix.values[i,j] < 0:
                    _d = -1
                else:
                    _d = 1
                _val = np.abs(adjacency_matrix.values[i,j]) ** beta
                adjacency_matrix.values[i,j] = _val * _d
                if adjacency_matrix.values[i,j] < threshold:
                    adjacency_matrix.values[i,j] = 0
                    

            
        return adjacency_matrix
    
    def _filter(self):
        if not self.D.index.equals(self.F.index):
            raise ValueError("invalid index for D and F")


        selected_genes_set = set()
    

        for fate in self.F.columns:

            correlations = self.D.corrwith(self.F[fate], method=self.method)
        
            top_genes_for_fate = correlations.nlargest(self.top_n).index.tolist()
        
            selected_genes_set.update(top_genes_for_fate)
        
        final_gene_list = sorted(list(selected_genes_set))
    
        D_prime = self.D[final_gene_list]     

        return D_prime,final_gene_list
   