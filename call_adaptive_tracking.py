import os
from adaptive_tracking import adaptive_tracking
import multiprocessing as mp


def mpro(de, u, f0, asc):
    os.environ["OMP_NUM_THREADS"] = "%s" % 2
    adaptive_tracking(3, de, u, f0, asc)


if __name__ == "__main__":
    Us = [0]
    deltas = [0.05]
    ascales = [1.001]
    F0s = [3, 10]
    jobs = []
    njobs = 0
    print("Starting multiprocessing procedure")
    for k in range(0, len(Us)):
        for el in range(0, len(deltas)):
            for m in range(0, len(ascales)):
                for g in (range(0, len(F0s))):
                    U = Us[k]
                    delt = deltas[el]
                    ascale = ascales[m]
                    F0 = F0s[g]
                    process = mp.Process(target=mpro, args=[delt, U, F0, ascale])
                    njobs += 3
                    if njobs > 72:
                        print("Max number of cores used")
                        break
                    process.start()
                    jobs.append(process)

    for j in jobs:
        j.join()