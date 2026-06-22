# ml_clinical_trial  
A machine learning framework for clinical trial data analysis.

---

## Prerequisites

Before working with the **ml_clinical_trial** framework, ensure that your system meets the following requirements:

### 1. System Requirements
- **Operating System**: Linux-based environment (tested on CentOS/Ubuntu HPC systems, the framework is developed on HiPerGator, University of Florida)
- **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).
- **SLURM**: Available for submitting jobs (required for `.sbatch` scripts)

### 2. HPG VSCode Remote Tunnels Setup

To develop on HiPerGator with full VS Code features, use **VSCode Remote Tunnels** instead of direct SSH.

Follow the official UF Research Computing guide:  
[HPG VSCode Remote Tunnels Setup](https://docs.rc.ufl.edu/domain/vscode_development/)

**Benefits:**
- Avoids account suspension from login node misuse.
- Full IntelliSense, debugging, and Git integration.
- Runs on allocated compute nodes via Slurm jobs.

**Tip:** Have your raw data and project code ready on HiPerGator before starting the tunnel.


---

## Handling Missing Values

In this repository, missing values in the dataset are represented using:

```python
import numpy as np
np.nan
```

---

## Set Up Workspace

Create **`j_map_2025_8_14`** project root folder:
```bash
mkdir /path/to/project/folder/j_map_2025_8_14
```

Change to **`j_map_2025_8_14`** project root folder:
```bash
cd /path/to/project/folder/j_map_2025_8_14
```
Git clone the repository branch to the project root folder

Change to the folder above ml_clinical_trial
```bash
cd /path/to/project/folder/j_map_2025_8_14/ml_clinical_trial
```
---
## GitHub Repository Documentation

### Permission Configuration of Bash Scripts
Purpose: Check and change the permission of bash scripts
**Note**: add 'sudo' before each lines of these two commands if required
To add permission to run the bash scripts, run:
```bash
chmod +x ./edit_git_config_in_script.sh
chmod +x ./update_github_branch.sh
```

### Update Git Configuration
Purpose: Registration of user's name, email, branch to trace and log the modification of repository changes automatically.
To set your Git user name and email in the script, run:

```bash
./edit_git_config_in_script.sh "your_name" "your.email@example.com" "branch_name"
```

This updates the Git configuration lines in `update_github_branch.sh`.

### Push Git Update

**Note**: Finish **Update Git Configuration** before this operation
To push your revision to your git branch, run:

```bash
./update_github_branch.sh "your_comments"
```

---

## Set Up the Conda Environment

This project uses a reproducible **Conda environment** to manage dependencies.
You can install it either **via HPC / SLURM (recommended)** or **locally**.


### 1. HPC / SLURM Setup (Recommended)

If you’re running on an HPC cluster with SLURM, use the provided batch script to install or update the environment automatically.

```bash
sbatch install_env.sbatch /path/to/environment.yml env_name
# $1 = /path/to/environment.yml → the path to the conda environment YAML file
# $2 = env_name               → the name of the conda environment to create
```

This will:

* Create a timestamped log directory:
  `installation_log_YYYY_MM_DD_HH_MM/`
* Build or update the environment named **`clinical_ml`**.
* Record all installation logs for debugging.

After the job finishes, activate the environment:

```bash
conda activate env_name
# $1 = env_name               → the name of the conda environment to create
```


### 2. Local Setup (Alternative)

If you prefer to install locally, follow these steps:

#### a. Check `conda` Installation

```bash
conda --version
```

#### b. Create the Environment

```bash
conda env create -f environment.yml -n env_name
# -f environment.yml → specifies the environment definition file
# -n env_name        → the name of the conda environment to create
```

#### c. Activate the Environment

```bash
conda activate env_name
# $1 = env_name               → the name of the conda environment to create
```

#### d. Update the Environment

When `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
# -f environment.yml → specifies the environment definition file
```

(`--prune` removes packages no longer needed.)


### 3. Sharing the Environment

If you add new packages, update the YAML so others can reproduce your setup:

```bash
conda env export > environment.yml
# environment.yml → specifies the environment definition file
```

Commit and push this updated file to the repository.


### Summary

* **Use SLURM for reproducible, logged installs on HPC.**
* **Use Conda locally if running on your own machine.**
* Keep `environment.yml` up to date for team consistency.

---

## Creating a Local Dataset with `jmap_act_data_processor`

To generate a local dataset for the ML clinical trial workflows, use the `jmap_act_data_processor` module.  
From the **`j_map_2025_8_14`** project root, run:

```bash

conda activate env_name
# $1 = env_name               → the name of the conda environment to create

python -m ml_clinical_trial.ml_clinical_act_jmap.jmap_act_data_processor
````

This command imports the J-MAP ACT data in MNI space registrated by SPM12 defined in the `jmap_act_data_processor` script and outputs the prepared data and data links to the designated local data directory.
Ensure that any necessary input files are available in the expected locations before running the command: `/orange/ruogu.fang/data/act_jmap_in_mni_by_spm12_junfu_cheng`


---

## Run the Experiment

To start the experiment, submit the SLURM job:

```bash
sbatch /path/to/your/workspace/run_ml_clinical_mini.sbatch /path/to/your/workspace/
```

This executes the training or analysis workflow defined in your experiment script.


