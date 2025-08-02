#!/bin/python
import numpy as np
import pandas as pd
import pickle
import os 
import sys
import itertools
from tqdm import tqdm
from dag import grn


# load experiment info file and point to results
grn_file = sys.argv[1]
out_stub = os.path.join(os.path.dirname(grn_file),
                        'graphs',
                        os.path.splitext(os.path.basename(grn_file))[0])
grns = pd.read_parquet(grn_file)


# helper extraction function(s)
def count_triangles(G):
    return sum(G.subgraph(G.predecessors(n)).number_of_edges() for n in G.nodes())

def count_diamonds(G):
    return sum(len(set(G.predecessors(p1)) & set(G.predecessors(p2))) 
               for n in G.nodes() 
               for p1,p2 in itertools.combinations(G.predecessors(n), 2)
              )

def extract_stats(g_file):
    with open(g_file, 'rb') as g:
        G = pickle.load(g) 
    return G.cish2, (G.L**2 - np.eye(G.n)).max(axis=0) / np.diag(G.L.T.dot(G.L)), count_triangles(G), count_diamonds(G)


# extract data: cis heritabilities, motif counts, and compute a median
stats = list(map(extract_stats, tqdm(['.'.join((out_stub, str(row.name), 'pkl')) for _,row in grns.iterrows()])))

grns[r'$V_{cis}/V_{tot}$'] = list(map(lambda x:x[0], stats))
grns[r'$max(B_{trans}^2/V_{cis})$'] = list(map(lambda x:x[1], stats))
grns[['triangles', 'diamonds']] = list(map(lambda x:x[2:4], stats))
grns['Median '+r'$V_{cis}/V_{tot}$'] = grns[r'$V_{cis}/V_{tot}$'].apply(np.median)

# done!
grns.to_parquet(grn_file)
