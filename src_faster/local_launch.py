import os
import re

import numpy as np
from glob import glob

from subprocess import Popen
import sys
import time

os.makedirs("logs",exist_ok=True)

blocking=True

# =========================== Experiment Choice ================== #
seed=np.arange(0,500,2)
frame_skip=[4]
dt_ckf=[0.025]
dt_control=[0.1]
N_radar=[3]
N_steps=[600]
move_radars = ["move_radars","no-move_radars"]
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
num_traj=[200] # 250
MPPI_iterations=[5] # 25

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
alpha3=[500]
alpha4=[1]
alpha5=[0]

import GPUtil
multi_device=True
fim_methods = ["SFIM","SFIM_bad","PFIM","PFIM_bad"]

for fim_method in fim_methods:
    for move_radar in move_radars:
        for seed_i in seed:
            for n_radar in N_radar:
                for n_traj in num_traj:
                    for mppi_iter in MPPI_iterations:
                        experiment_name = os.path.join(f"Experiment11_{fim_method}",f"N_radar={n_radar}-{move_radar}")
                        results_savepath = "results"
                        for n_steps in N_steps:
                            file = f"--{move_radar} " \
                                f"--seed={seed_i} " \
                                f"--experiment_name={experiment_name} " \
                                f"--results_savepath={results_savepath} " \
                                f"--N_radar={n_radar} " \
                                f"--N_steps={n_steps} " \
                                f"--no-save_images " \
                                f"--MPPI_iterations={mppi_iter} " \
                                f"--num_traj={n_traj} " \
                                f"--fim_method={fim_method} " \
                                f"--R={R[0]} "

                            filepath = os.path.join(results_savepath,experiment_name+f"_{seed_i}")
                            # print(filepath)
                            rmse_exists = len(glob(os.path.join(filepath, "*rmse*"))) >= 1

                            if os.path.exists(filepath) and rmse_exists:
                                print(filepath,"exists")
                                continue


                            if multi_device:
                                deviceIDs = GPUtil.getAvailable(order = 'first', limit = 4, maxLoad = 0.8, maxMemory = 0.15, includeNan=False, excludeID=[3], excludeUUID=[])
                                print(deviceIDs)
                                file_full = f"python main_expectation.py {file}"
                                file_run = os.path.join(os.getcwd(),"execute_local.bash")
                                if len(deviceIDs) > 0:
                                    print(f"GPU Device {deviceIDs[0]}")
                                    print(f"tmux new-session -d {file_run} '{file_full}' '{deviceIDs[0]}'")
                                    Popen(f"tmux new-session -d bash {file_run} '{file_full}' '{deviceIDs[0]}'",shell=True) #, shell=True,creationflags=CREATE_NEW_CONSOLE)
                                    time.sleep(7)
                                else:
                                    print("All GPUs USED AT THIS MOMENT, WAIT UNTIL NEW RESOURCE AVAILABLE")
                                    while len(deviceIDs) == 0:
                                        deviceIDs = GPUtil.getAvailable(order = 'first', limit = 4, maxLoad = 0.8, maxMemory = 0.15, includeNan=False, excludeID=[3], excludeUUID=[])
                                        time.sleep(5)
                                    print(f"GPU Device {deviceIDs[0]}")
                                    print(f"tmux new-session -d {file_run} '{file_full}' '{deviceIDs[0]}'")
                                    Popen(f"tmux new-session -d bash {file_run} '{file_full}' '{deviceIDs[0]}'",shell=True) #, shell=True,creationflags=CREATE_NEW_CONSOLE)
                                    time.sleep(7)
                            else:
                                file_full = f"python main_expectation.py {file}"
                                file_run = os.path.join(os.getcwd(),"execute_local.bash")
                                print(file_run)
                                os.system(f"bash {file_run} '{file_full}' '{0}'")