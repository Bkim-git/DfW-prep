import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt

working_dir = "Specify the directory containing the runtfile"
os.chdir(working_dir)

import shutil
from modules import *
import gc

def main():
    Cases = ["Example"]
    for Case in Cases:
        DfW_prep(Case)
        gc.collect()

def DfW_prep(Case):
    params = Load(Case)
    vid = Orthorectification(params).run()
    gcps_px = Track_GCPs(vid, params).run(replace=False)
    vid_out, extent = Georeferencing(params, gcps_px).warp_batch(vid)

    Visualization(vid_out, extent, params)
    CoRegistration(vid_out, extent, params).run()

if __name__ == "__main__":
    main()
