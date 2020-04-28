import os
from main_tracking import main_track
import multiprocessing as mp


def mpro(u, de, asc, f0):
    os.environ["OMP_NUM_THREADS"] = "%s" % 2
    main_track(3, 6, u, de, asc, f0)


if __name__ == "__main__":
    Us = [0]
    deltas = [0.01]
    ascales = [1.0001]
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
                    process = mp.Process(target=mpro, args=[U, delt, ascale, F0])
                    njobs += 3
                    if njobs > 72:
                        print("Max number of cores used")
                        break
                    process.start()
                    jobs.append(process)

    for j in jobs:
        j.join()

    print("Tracking Sim finished.")
