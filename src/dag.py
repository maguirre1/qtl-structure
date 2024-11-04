#!/bin/python3
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as ss


class grn(nx.DiGraph):
    def __init__(self):
        super(grn, self).__init__()


    def add_structure(self, method, n_genes, **kwargs):
        # initialize a graph structure into the below adjacency matrix
        self.A = np.empty((n_genes, n_genes))
        self.n = n_genes
        
        # assign genes to groups
        self.I = np.empty(n_genes)
        
        # ER: Erdos-Renyi 
        if method == "er":
            self.add_er_structure(n_genes, **kwargs)
        
        # PPM: Planted Partition Model
        elif method == "ppm":
            self.add_ppm_structure(n_genes, **kwargs)
        
        # DSFG: Directed Scale-free Graph
        elif method == "dsfg":
            self.add_dsfg_structure(n_genes, **kwargs)
        
        else:
            raise ValueError("method must be one of: er, ppm, dsfg")
            
        # add structure into the networkx object
        self.add_edges_from(zip(*np.where(self.A)))
        
        return self
    
    
    def add_er_structure(self, n_genes, p=None, r=None):
        # check for one of the following parameters: 
        # - p is probability of edge existing
        # - r is expected regulators per gene
        if p is None:
            if r is None:
                raise ValueError("p and r cannot both be None")
                return
            p = 2*float(r)/(n_genes - 1)
        
        self.A = np.triu(np.random.binomial(n = 1, p = p, size = (n_genes, n_genes)), k = 1).astype(int)
        self.I = np.ones(n_genes)
        self.n = n_genes
        return self
    
    
    def add_ppm_structure(self, n_genes, n_groups, m=None, r=None, p=None, q=None, hierarchy=True, seed=None):
        # check for one combination of the following parameters:
        # - p, q: probability of within- and between-group edges
        # - r, m: expected regulators per gene, expected fraction of edges within-group
        k = n_groups
        # check for being lazy:
        if n_groups == 1:
            return self.add_er_structure(n_genes, p, r)
        # now do this
        if p is None or q is None:
            if r is None or m is None:
                raise ValueError("One of (p and q) or (r and m) must be specified")
            P = 2 * k * r * ( (1-m) * np.ones((k,k)) + (m*k - 1)*np.eye(k) ) / ((k-1) * (n_genes-1))
        else:
            P = (p * np.eye(k)) + ((q-p) * np.ones((k,k)))
        
        # graph structure
        sizes = [int(n_genes / n_groups) + int(i < n_genes % n_groups) for i in range(n_groups)]
        A = nx.to_numpy_array(
                nx.stochastic_block_model(
                    sizes = sizes,
                    p = P,
                    seed = seed
                )
        )
                
        # group membership
        I = np.array([i for i,size in enumerate(sizes) for _ in range(size)])
        
        # ordering -- this could be done in a separate function
        if hierarchy:
            # groups determine topological sort (this is the networkx default)
            self.A = np.triu(A, k=1)
            self.I = I
        else:
            # we need to impose a random topological ordering
            order = np.random.choice(np.arange(n_genes), size = n_genes, replace = False)
            self.A = np.triu(A[np.ix_(order, order)], k = 1)
            self.I = I[order]
        self.n = n_genes
            
        return self
    
    
    def add_dsfg_structure(self, n_genes, n_groups, r=5, d=10, w=1, hierarchy=True):
        ## TODO
        self.n = n_genes
        
        # group assignment
        sizes = [int(n_genes / n_groups) + int(i < n_genes % n_groups) for i in range(n_groups)]
        self.I = np.array([i for i,size in enumerate(sizes) for _ in range(size)])
        if not hierarchy:
            np.random.shuffle(self.I)
        
        # for the top of the graph: assign linear structure for preferential attachment
        self.A = np.zeros((n_genes, n_genes))
        self.A[:int(2*r), :int(2*r)] = np.eye(int(2*r))
        
        # assign each gene a poisson number of regulators centered at r
        #  with incoming rates proportional to w if in the same group, otherwise 1
        #  and with preferential attachment on out-degree, with linear biasing term d
        for i in range(int(2*r), self.n):
            # pick parents
            wts = (1 + (w-1)*(self.I[:i]==self.I[i]) ) * (self.A[:i,:i].sum(axis=1) + d)
            for j in np.random.choice(np.arange(i), 
                                      size = np.random.poisson(r), 
                                      p = wts / np.sum(wts), 
                                      replace = True):
                self.A[j,i] = self.A[j,i] + 1   
        
        # actually set the top of the graph to have E-R structure
        self.A[:int(2*r), :int(2*r)] = np.triu(np.random.binomial(n = 1, 
                                                                  p = 2 * r / (n_genes-1), 
                                                                  size = (int(2*r), int(2*r))
                                                                 ), 
                                               k = 1
                                              ).astype(int)       
        
        return self
    
    
    def add_expression_parameters(self, sign, weight):
        # parameters:
        # - sign (n,) or (n,n): how to assign sign to edges
        # - weight (n,) or (n,n): how to assign weights to edges
        self.A = sign * (weight * self.A.astype(bool).astype(int) )
        
        # update nx stuff
        self.add_edges_from(zip(*np.where(self.A)), weight=self.A[np.where(self.A)])
        
        return self
    
    
    def compute_h2(self, V_cis=None):
        # computes gene covariance L as
        #  L = (V_cis x I)(I - A)^{-1}
        # computes cis contribution to heritability as 
        #  V_cis / V_tot = (L x I)^2 / ((L^T L) x I)
        if V_cis is None and ('V_cis' not in dir(self) or self.V_cis is None):
            self.V_cis = np.ones(self.n)
        else:
            self.V_cis = V_cis
        
        self.L = np.diag(np.sqrt(self.V_cis)) @ np.linalg.pinv(np.eye(self.n) - self.A)
        
        self.cish2 = np.diag(self.L)**2 / np.diag(self.L.T.dot(self.L))
        
        return self
    
    
    def simulate_population(self):
        ## TODO
        return self