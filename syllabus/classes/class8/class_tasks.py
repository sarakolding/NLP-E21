import gensim.downloader as api
model = api.load("glove-wiki-gigaword-50")
import numpy as np

cloud = model["cloud"]
table = model["table"]
chair = model["chair"]

print(chair @ table)

print(chair @ cloud)
print(table @ cloud)

matrix = np.array([cloud, table, chair])

matrix_t = matrix.transpose()

matrix_mult = matrix @ matrix_t
print(matrix_mult)