"""
Multi-device matrix multiplication using parla with cupy as the kernel engine.

"""
import sys
import time
import os


import argparse

parser = argparse.ArgumentParser()
#Size of matrix
parser.add_argument('-n', type=int, default=4, help='Size of matrix')
#How many trials to run
parser.add_argument('-trials', type=int, default=1, help='Number of trials to run')
#Are the placement fixed by the user or determed by the scheduler?
parser.add_argument('-fixed', default=0, type=int, help="User Mapping (1) or Scheduler Mapping (0)")
#Number of devices to use
parser.add_argument('-ngpus', type=int, default=1, help="Number of GPUs to use")
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

import numpy as np
import cupy as cp

from parla import Parla, get_all_devices
from parla.array import copy, clone_here
from parla.cpu import cpu
from parla.cuda import gpu
from parla.function_decorators import specialized
from parla.ldevice import LDeviceSequenceBlocked
from parla.tasks import spawn, TaskSpace, CompletedTaskSpace, reserve_persistent_memory
from parla.parray import asarray_batch
import math

h_ordr = 'C'
d_ordr = 'F'
n = args.n

# Function to calculate the softmax of the matrix    
def soft_max(cuarray):
    e_x = np.exp(cuarray)
    #print("Serial sum", e_x.sum())
    return e_x / e_x.sum()

# Function to calculate the attention score serially
def original_attention(queries, keys, values, sqrt_d_k):
    attention_scores = queries @ keys.T
    norm_attention_scores = attention_scores/sqrt_d_k
    #print("step 1 matrics = ", norm_attention_scores)
    attention_weights = soft_max(norm_attention_scores)
    ret_val = attention_weights @ values
    #print("step 2 matrics = ", ret_val)
    return ret_val

def main(a_cpu, b_cpu, c_cpu, sqr_d_k):
    verbose = True
    @spawn(placement=cpu)
    async def main_task():
        ngpus = cp.cuda.runtime.getDeviceCount()
        repetitions = args.trials

        # set up two n x n arrays to multiply together.
        # n is chosen so that all three can be
        # stored within the memory of a single GPU
        # so that strong scaling numbers make sense.
        

        blocks = ngpus
        block_size = n // ngpus
        print("ngpus", ngpus)
 
        #print("BlockSize: ", block_size, "GPUS: ", ngpus)

        # Overdecomposing doesn't actually seem to help in this case
        # with the current parla runtime. This may be related to
        # some weirdness within the scheduler though, so
        # we can leave the code for blocks in-place for further
        # testing later.

        
        #print("Finished Data Allocation", flush=True)
        # Partition the two arrays and set up the
        # partitioned array where the result will be stored.
        # This could also be done using a parla mapper object.

        a_part = []
        b_part = []
        c_part = []
        
        ab_part = []
        ab_softmax_part = []

        distribute=True
        reset=True
        fixed_placement=args.fixed
        verbose=False
        sync=True

        time_list = list()

        # Start all operans from CPU memory.
        for i in range(blocks):
            if distribute:
                with cp.cuda.Device(i):
                    a_part.append(cp.asarray(a_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
                    b_part.append(cp.asarray(b_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
                    c_part.append(cp.asarray(c_cpu[i * block_size : (i + 1) * block_size], order=d_ordr))
                    cp.cuda.stream.get_current_stream().synchronize()
            else:
                a_part.append(a_cpu[i * block_size : (i + 1) * block_size])
                b_part.append(b_cpu[i * block_size : (i + 1) * block_size])
                c_part.append(c_cpu[i * block_size : (i + 1) * block_size])

        for i in range(blocks):
            ab_part.append(list())
            ab_softmax_part.append(list())
            for j in range(blocks):
                ab_part[i].append(np.empty((0, 0), dtype=np.float32, order=h_ordr))
                ab_softmax_part[i].append(np.empty((0, 0), dtype=np.float32, order=h_ordr))

        #print(len(ab_part), len(ab_part[0]), ab_part[0][0].shape)

        # 1. NEW: convert to parray in batch
        a_part, b_part = asarray_batch(a_part, b_part)
        c_part = asarray_batch(c_part)
        ab_part = asarray_batch(ab_part)
        ab_softmax_part = asarray_batch(ab_softmax_part)

        #print(len(ab_part), len(ab_part[0]))


        for repetition in range(repetitions):
            '''	
            #reset cblocks to None
            for i in range(blocks):
                for j in range(blocks):
                    ab_part[i][j].update(np.empty((0, 0), dtype=np.float32, order=h_ordr))
                    ab_softmax_part[i][j].update(np.empty((0, 0), dtype=np.float32, order=h_ordr))

            if reset:
                #reset coherence to only be in starting locations
                rspace = TaskSpace("reset")
                for i in range(blocks):
                    @spawn(rspace[i], placement=gpu(i%ngpus), memory=2*block_size*n, inout=[a_part[i], b_part[i], c_part[i]])
                    def reset_task():
                        a_part[i].update(a_part[i].array)
                        b_part[i].update(b_part[i].array)
                        c_part[i].update(c_part[i].array)
                await rspace
            '''

            matmul = TaskSpace("matmul")
            e_sums = np.zeros(1)
            #To store partial sums for all the blocks
            e_array = np.zeros(blocks*blocks)
            start = time.perf_counter()
            for i in range(blocks):
                for j in range(blocks):
                    a_block = a_part[i]
                    b_block = b_part[j]
                    ab_block = ab_part[i][j]
                    memsize = (block_size**2)*4

                    if fixed_placement:
                        loc = gpu(i%ngpus)
                    else:
                        loc = gpu

                    @spawn(matmul[i, j], placement = loc, memory=memsize, input=[a_block, b_block], output=[ab_block])
                    def matmul_task():
                        a = a_block.array
                        b = b_block.array
                        c = ab_block.array
                        #print(len(a), len(b))
                        stream = cp.cuda.get_current_stream()
                        stream.synchronize()

                        assert(a.device.id == b.device.id)
                        if verbose:
                            print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", cp.cuda.runtime.getDevice(), a.device.id, flush=True)
                        local_start = time.perf_counter()
                        #print(len(a), len(a[0]), len(b), len(b[0]))
                        c = cp.matmul(a,b.T)
                        #Dividing c with sqr_d_k here
                        c = c/sqr_d_k
                        #using e_sums to accumulate cp.exp(c)
                        e_sums = cp.exp(c).sum(keepdims=True)
                        e_array[i*blocks + j] = e_sums

                        if sync:
                            stream.synchronize()
                        local_end = time.perf_counter()

                        ab_block.update(c)
                        c = ab_block.array

                        if verbose:
                            print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)

            await matmul
            #Finally reducing the array of partial sums to a single value
            e_sum_final = e_array.sum()
            print("Parallel Sum: ", e_sum_final)

            #print("Step 1 Parla: ")
            #for row in ab_part:
            #    for col in row:
            #        print(col)

            # The Output matrix is in blocks, and the input matrix needs to be in block-row form
            # Hence changing the layout the ab matrix
            ab_block_row = []
            matmult2 = TaskSpace("matmul2")
            for i in range(blocks):
              	# Accesing the row (which has blocks) 
                ab_block_row = ab_part[i]
                memsize = (block_size**2)*4

                if fixed_placement:
                    loc = gpu(i%ngpus)
                else:
                    loc = gpu

                @spawn(matmult2[i], placement=loc)
                def matmul2_task():
					# Accesing the first block in the selected row
                    ab_strip = ab_block_row[0].array
                    for j in range(1,blocks):
						# placing the other blocks in the same row
                        ab_next = ab_block_row[j].array
                        ab_strip = cp.concatenate((ab_strip, ab_next), axis=1)
					# finally updating the pointer
                    ab_block_row[0].update(ab_strip)

                    stream = cp.cuda.get_current_stream()
                    stream.synchronize()
            await matmult2

            # TODO: You can calculate Softmax(AB/d)*C here
            matmult3 = TaskSpace("matmul3")
            for i in range(blocks):
                for j in range(blocks):
					# Acessing the strip in rows
                    ab_block = ab_part[i][0]
                    c_block = c_part[j]
                    ab_softmax_block = ab_softmax_part[i][j]
                    memsize = (block_size**2)*4

                    if fixed_placement:
                        loc = gpu(i%ngpus)
                    else:
                        loc = gpu

                    @spawn(matmult3[i, j], placement=loc, memory=memsize, input=[ab_block, c_block], output=[ab_softmax_block])
                    def matmul3_task():
                        ab = ab_block.array #score
                        c = c_block.array   #value
                        ab_softmax = ab_softmax_block.array
                        stream = cp.cuda.get_current_stream()
                        stream.synchronize()

                        assert(ab.device.id == c.device.id)
                        if verbose:
                            print(f"+({i}, {j}): ", ab.shape, c.shape, ab_softmax.shape, " | On Device: ", cp.cuda.runtime.getDevice(), ab.device.id, flush=True)
                        local_start = time.perf_counter()
                        ab = cp.exp(ab) / e_sum_final
                        ab = cp.exp(ab)
                        #print(len(ab), len(ab[0]))

						#Finally calculating the attention score
                        ab_softmax = cp.matmul(ab,c.T) #attention
                        if sync:
                            stream.synchronize()
                        local_end = time.perf_counter()

                        ab_softmax_block.update(ab_softmax)
                        ab_softmax = ab_softmax_block.array
                        #print(ab_softmax)
                        if verbose:
                            print(f"-({i}, {j}): ", ab.shape, c.shape, ab_softmax.shape, " | Elapsed: ", local_end-local_start, flush=True)

            await matmult3
            #print("Step 2 Parla: ")
            #for row in ab_softmax_part:
            #    for col in row:
            #       print(col)
            stop = time.perf_counter()
            #print(f"Iteration {repetition} | Time:", stop - start, flush=True)
            time_list.append(stop-start)
            
            
        mean = np.mean(np.array(time_list))
        median = np.median(np.array(time_list))
        print(f"Execution:: Average = {mean} | Median = {median}", flush=True)

if __name__ == "__main__":
    d_k = n
    d_v = n
    input_seq_length = n
    
    np.random.seed(0)
    a_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)
    b_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)
    c_cpu = np.random.rand(n, n).astype(dtype=np.float32, order=h_ordr)
    
    queries = cp.asarray(a_cpu)
    keys = cp.asarray(b_cpu)
    values = cp.asarray(c_cpu)


    s1 = time.perf_counter()
    cc = original_attention(queries, keys, values, math.sqrt(d_k))
    s2 = time.perf_counter()
    print("Normal took ", s2-s1)
    
    verbose = True
 
    with Parla():
        main(a_cpu, b_cpu, c_cpu.T, math.sqrt(d_k))
       
