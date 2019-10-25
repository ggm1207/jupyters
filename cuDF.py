
import cudf
import numpy as np
import pandas as pd
from numba import cuda
import time
 
data_length = int(5e8)
average_window = 4
df = cudf.DataFrame()
threads_per_block = 128
trunk_size = 10240
df['in1'] = np.arange(data_length, dtype=np.float64)
 
 
def kernel1(in1, out, average_length):
    for i in range(cuda.threadIdx.x,
               average_length-1, cuda.blockDim.x):
        out[i] = np.inf
    for i in range(cuda.threadIdx.x + average_length - 1,
                   in1.size, cuda.blockDim.x):
        summ = 0.0
        for j in range(i - average_length + 1,
                       i + 1):
            summ += in1[j]
        out[i] = summ / np.float64(average_length)
 
def kernel2(in1, out, average_length):
    if in1.size - average_length + cuda.threadIdx.x - average_length + 1 < 0 :
        return
    for i in range(in1.size - average_length + cuda.threadIdx.x,
                   in1.size, cuda.blockDim.x):
        summ = 0.0
        for j in range(i - average_length + 1,
                       i + 1):
            #print(i,j, in1.size)
            summ += in1[j]
        out[i] = summ / np.float64(average_length)
 
 
start = time.time()
df = df.apply_chunks(kernel1,
                     incols=['in1'],
                     outcols=dict(out=np.float64),
                     kwargs=dict(average_length=average_window),
                     chunks=list(range(0, data_length,
                                       trunk_size))+ [data_length],
                     tpb=threads_per_block)
 
df = df.apply_chunks(kernel2,
                     incols=['in1', 'out'],
                     outcols=dict(),
                     kwargs=dict(average_length=average_window),
                     chunks=[0]+list(range(average_window, data_length,
                                           trunk_size))+ [data_length],
                     tpb=threads_per_block)
end = time.time()
print('cuDF time', end-start)
 
pdf = pd.DataFrame()
pdf['in1'] = np.arange(data_length, dtype=np.float64)
start = time.time()
pdf['out'] = pdf.rolling(average_window).mean()
end = time.time()
print('pandas time', end-start)
 
assert(np.isclose(pdf.out.as_matrix()[average_window:].mean(),
       df.out.to_array()[average_window:].mean()))