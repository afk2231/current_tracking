import os
import multiprocessing as mp
from work_phi_dot_tracking import work_phi_dot_tracking as wpdt

def mpro(f0):
    os.environ["OMP_NUM_THREADS"] = "%s" % 3
    wpdt(f0)



if __name__ == "__main__":
    njobs = 0
    jobs = []
    F0s = [3, 10]
    print("Starting multiprocessing procedure")
    for m in range(len(F0s)):
        F0 = F0s[m]
        process = mp.Process(target=mpro, args=[F0])
        njobs += 1
        if njobs > 72:
            print("Max number of cores used")
            break
        process.start()
        jobs.append(process)

    for j in jobs:
        j.join()

    print("Main Sim finished.")