import os
from main import main_sim
import multiprocessing as mp
import numpy as np


def mpro(u, de, f0):
    os.environ["OMP_NUM_THREADS"] = "%s" % 1
    main_sim(3, 6, u, de, f0)


if __name__ == "__main__":
    Us = [0]
    deltas = [0.01]
    F0s = [3, 10]
    jobs = []
    njobs = 0
    print("Starting multiprocessing procedure")
    for k in range(0, len(Us)):
        for el in range(len(deltas)):
            for m in range(len(F0s)):
                U = Us[k]
                delt = deltas[el]
                F0 = F0s[m]
                process = mp.Process(target=mpro, args=[U, delt, F0])
                njobs += 1
                if njobs > 72:
                    print("Max number of cores used")
                    break
                process.start()
                jobs.append(process)

    for j in jobs:
        j.join()

    print("Main Sim finished.")
