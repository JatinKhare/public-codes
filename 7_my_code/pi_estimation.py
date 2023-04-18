from parla import Parla
# Import for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace
import time
import numpy as np
from parla.cpu import cpu
import cupy
from parla.cuda import gpu
from numba import jit, njit
import random
import argparse
# importing functools for reduce()
import functools

parser = argparse.ArgumentParser()
parser.add_argument('-trials', type=int, default=1000, help='Number of trials to run')
parser.add_argument('-tasks', type=int, default=2, help='Number of tasks for Parla')
args = parser.parse_args()

INTERVAL = args.trials
Tasks = args.tasks

@jit("boolean()", fastmath=True, nogil=True, nopython=True)
def one_sample():
    rand_x = random.uniform(-1, 1)
    rand_y = random.uniform(-1, 1)
    origin_dist = 0 
    # Distance between (x, y) from the origin
    # TODO: calculate the origin_dist
    origin_dist = rand_x*rand_x + rand_y*rand_y

    if origin_dist <= 1:
        return True
    else:
        return False

@jit("i4(i4)", fastmath=True, nogil=True, nopython=True)
def estimate_pi(length):
    circle_points = 0
    for index in range(length):
        if one_sample():
            circle_points += 1
    return circle_points


@jit("i4(i4, i4)", fastmath=True, nogil=True, nopython=True)
def parla_estimate_pi(length, task):
    circle_points = 0
    for index in range(int(length/task)):
        if one_sample():
            circle_points += 1
    return circle_points

def Parla_PI():
    @spawn()
    async def main_task():
        ID = TaskSpace(Tasks)
        for i_task in range(Tasks):
            @spawn(ID[i_task])
            async def parla_function():
                arr_num[i_task] = parla_estimate_pi(INTERVAL, Tasks)
        await parla_function 
        final_sum = sum(arr_num)
        pi = 4 * final_sum / INTERVAL
        print("Parla final estimation of Pi =", pi)

       
if __name__ == "__main__":
    t = time.time()
    circle_points = estimate_pi(INTERVAL)
    pi = 4 * circle_points / INTERVAL
    elapsed = time.time() - t
    print("Python final estimation of Pi =", pi)
    print("Original elapsed time: ", elapsed)
    orig_elapsed = elapsed
    t = time.time()        
# New array to store the partial sums from each task
    arr_num = [0] * Tasks
    with Parla():
        Parla_PI()
    elapsed = time.time() - t
    print("PARLA elapsed time: ", elapsed)
    print("Speed up:", orig_elapsed/elapsed)
