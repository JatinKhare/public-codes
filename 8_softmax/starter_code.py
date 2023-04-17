"""
Multi-device matrix multiplication using parla with cupy as the kernel engine.

"""
import sys
import time
import os


import argparse

parser = argparse.ArgumentParser()
#Size of matrix
parser.add_argument('-n', type=int, default=32000, help='Size of matrix')
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
    
    
def soft_max(cuarray):
    #finish this function
    pass

def original_attention(queries, keys, values, sqrt_d_k):
    # Finish this function
    pass
    
def main(a_cpu, b_cpu, c_cpu, sqr_d_k):

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

                        stream = cp.cuda.get_current_stream()
                        stream.synchronize()

                        assert(a.device.id == b.device.id)
                        if verbose:
                            print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", cp.cuda.runtime.getDevice(), a.device.id, flush=True)
                        local_start = time.perf_counter()
                        c = a @ b.T 
                        #TODO: You can divid the c to sqr_d_k here
                        # e_sums[0] can be used to accumulate cp.exp(c)
                        if sync:
                            stream.synchronize()
                        local_end = time.perf_counter()

                        ab_block.update(c)
                        c = ab_block.array

                        if verbose:
                            print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)

            await matmul
            
            ab_part = ab_part[0]
            #matmult2 = TaskSpace("matmul2")
            # TODO: You can clculate Softmax(AB/d)*C here
            matmul = TaskSpace("matmul")
            e_sums = np.zeros(1)
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

                        stream = cp.cuda.get_current_stream()
                        stream.synchronize()

                        assert(a.device.id == b.device.id)
                        if verbose:
                            print(f"+({i}, {j}): ", a.shape, b.shape, c.shape, " | On Device: ", cp.cuda.runtime.getDevice(), a.device.id, flush=True)
                        local_start = time.perf_counter()
                        c = a @ b.T 
                        #TODO: You can divid the c to sqr_d_k here
                        # e_sums[0] can be used to accumulate cp.exp(c)
                        if sync:
                            stream.synchronize()
                        local_end = time.perf_counter()

                        ab_block.update(c)
                        c = ab_block.array

                        if verbose:
                            print(f"-({i}, {j}): ", a.shape, b.shape, c.shape, " | Elapsed: ", local_end-local_start, flush=True)

            await matmul
                
            
            
            
            stop = time.perf_counter()
#            print(f"Iteration {repetition} | Time:", stop - start, flush=True)
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
    
    
    with Parla():
        main(a_cpu, b_cpu, c_cpu, math.sqrt(d_k))