import os
import re

import numpy as np
from glob import glob

from subprocess import Popen

os.makedirs("logs",exist_ok=True)

blocking=False
batch_command = "--job-name=freedom --exclusive --cpus-per-task=18 --mem=20Gb --partition=short"

# =========================== Experiment Choice ================== #
seed=np.arange(0,500,2)
frame_skip=[4]
dt_ckf=[0.025]
dt_control=[0.1]
N_radar=[3]
N_steps=[1000]
move_radars = ["no-move_radars","move_radars"]
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
                if os.path.exists(filepath):
                    print(filepath,"exists")
                    rmse_exists = len(glob(os.path.join(filepath,"*rmse*"))) >= 1
                    continue

                if blocking:
                    os.system(f"python main_expectation.py {file}")
                else:
                    file_full = f"python main_expectation.py {file}"
                    print(f"sbatch execute.bash '{file_full}'")
                    Popen(f"sbatch execute.bash '{file_full}'", shell=True)