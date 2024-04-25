import os
import re

import numpy as np
from glob import glob

from subprocess import Popen
import sys
import time

os.makedirs("logs",exist_ok=True)

blocking=False

# =========================== Experiment Choice ================== #
seed=np.arange(0,500,2)
frame_skip=[4]
dt_ckf=[0.025]
dt_control=[0.1]
N_radar=[3]
N_steps=[1000]
move_radars = ["move_radars"] #["no-move_radars","move_radars"]
remove_tmp_images = ["remove_tmp_images"]
save_images = ["no-save_images"]

# ==================== RADAR CONFIGURATION ======================== #
fc=[1e8]
Gt=[200]
Gr=[200]
rcs=[1]
L=[1]
R=[500]
Pt=[1000]
SNR=[-20]


# ==================== MPPI CONFIGURATION ======================== #
acc_std=[25]
ang_acc_std=[45*np.pi/180]
horizon=[15]
acc_init=[0]
ang_acc_init=[0 * np.pi/180]
num_traj=[250]
MPPI_iterations=[25]

# ==================== AIS  CONFIGURATION ======================== #
temperature=[0.1]
elite_threshold=[0.9]
AIS_method=["CE"]

# ============================ MPC Settings =====================================#
gamma =[0.95]
speed_minimum=[5]
R2T=[125]
R2R=[10]
alpha1=[1]
alpha2=[1000]
alpha3=[60]
alpha4=[1]
alpha5=[0]

import GPUtil

for move_radar in move_radars:
    for seed_i in seed:
        for n_radar in N_radar:
            experiment_name = os.path.join("experiment1_pcrlb_expectation",f"N_radar={n_radar}-{move_radar}")
            results_savepath = "results"
            for n_steps in N_steps:
                file = f"--{move_radar} " \
                       f"--seed={seed_i} " \
                       f"--experiment_name={experiment_name} " \
                       f"--results_savepath={results_savepath} " \
                       f"--N_radar={n_radar} " \
                       f"--N_steps={n_steps} " \
                       f"--fim_method='PCRLB'"

                filepath = os.path.join(results_savepath,experiment_name+f"_{seed_i}")
                # print(filepath)
                rmse_exists = len(glob(os.path.join(filepath, "*rmse*"))) >= 1

                if os.path.exists(filepath) and rmse_exists:
                    print(filepath,"exists")
                    continue

                deviceIDs = GPUtil.getAvailable(order = 'first', limit = 2, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
                print(deviceIDs)
                if len(deviceIDs) > 0:
                    file_full = f"python main_expectation.py {file}"
                    file_run = os.path.join(os.getcwd(),"execute_local.bash")
                    print(f"GPU Device {deviceIDs[0]}")
                    print(f"tmux new-session -d {file_run} '{file_full}' '{deviceIDs[0]}'")
                    Popen(f"tmux new-session -d bash {file_run} '{file_full}' '{deviceIDs[0]}'",shell=True) #, shell=True,creationflags=CREATE_NEW_CONSOLE)
                    time.sleep(30)
                else:
                    print("All GPUs USED AT THIS MOMENT, WAIT UNTIL NEW RESOURCE AVAILABLE")
                    while len(deviceIDs) == 0:
                        deviceIDs = GPUtil.getAvailable(order = 'first', limit = 2, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
                        time.sleep(250)