import argparse
from time import time

import hnswlib
import numpy as np
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--ef', type=int, default=200)
parser.add_argument('--ef_construction', type=int, default=200)
parser.add_argument('--m', type=int, default=16)
parser.add_argument('--n_queries', type=int, default=64)


args, unknown_args = parser.parse_known_args()  # Parses only the known args to fix an issue with argv[1] being used as a save path

num_elements = 100000
k = 1
num_queries = args.n_queries


# Generating sample data
data = []
dir = "/home/goji/Documents/PokemonRedRL/sessions/session_4a4b0522/novelty_frames"
_, _, files = next(os.walk(dir))
for i, file in enumerate(files):
    if "observed" in file:
        img = cv2.imread(dir + "/" + file, cv2.IMREAD_GRAYSCALE)
        data.append(img.flatten()[np.newaxis])
    print(i, len(files))

np.random.shuffle(data)
total_data = len(data)

data = np.concatenate(data, axis=0)

index_data, query_data = data[:-num_queries], data[-num_queries:]
# Declaring index
hnsw_index = hnswlib.Index(space='l2', dim=len(data[0]))  # possible options are l2, cosine or ip
bf_index = hnswlib.BFIndex(space='l2', dim=len(data[0]))

# Initing both hnsw and brute force indices
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# hnsw construction params:
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

hnsw_index.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.m)
bf_index.init_index(max_elements=num_elements)

# Controlling the recall for hnsw by setting ef:
# higher ef leads to better accuracy, but slower search
hnsw_index.set_ef(args.ef)

# Set number of threads used during batch search/construction in hnsw
# By default using all available cores
hnsw_index.set_num_threads(1)

print("Adding batch of %d elements" % (len(data)))
hnsw_index.add_items(index_data)
bf_index.add_items(index_data)

print("Indices built")

# Query the elements and measure recall:
t = time()

labels_hnsw, distances_hnsw = hnsw_index.knn_query(query_data, k)
labels_bf, distances_bf = bf_index.knn_query(query_data, k)

# Measure recall
correct = 0

for i in range(num_queries):
    for label in labels_hnsw[i]:
        for correct_label in labels_bf[i]:
            if label == correct_label:
                correct += 1
                break

dt = time() - t
print("recall is :", float(correct)/(k*num_queries))
print("time required:", dt)
print("time required per sample:", dt/num_queries)