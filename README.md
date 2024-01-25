# Introduction 
Welcome to the "NOTSOFAR-1: Distant Meeting Transcription with a Single Device" Challenge.

This repo contains the baseline system code for the NOTSOFAR-1 Challenge.

For more details see:
1. CHiME website: https://www.chimechallenge.org/current/task2/index
2. Preprint: https://arxiv.org/abs/2401.08887


# Project Setup
The following steps will guide you through setting up the project on your machine. <br>
This guide is written for Linux. Windows support is coming soon.

Alternatively, you can run and work on the project in a [devctonainer](https://containers.dev/) using, for example, the [Dev Containers VSCode Extension](https://code.visualstudio.com/docs/devcontainers/containers).

### Step 1: Clone the Repository

Clone the `NOTSOFAR1-Challenge` repository from GitHub. Open your terminal and run the following command:

```bash
sudo apt-get install git
git clone https://github.com/microsoft/NOTSOFAR1-Challenge.git
```
**Note**: this repository is still private, you will need to be added as a contributor to clone it. <br>

###### Cloning using Github Username
If you encounter an `Repository not found` error, try cloning the repository using the following command: <br>

```bash
git clone https://<github_username>@github.com/microsoft/NOTSOFAR1-Challenge.git
````

###### Cloning using Github Token
1. Create the token: <br>
   Go to GitHub Settings > Developer Settings > Personal Access Tokens > Generate New Token (classic) > Mark repo (with all sub items) > Click Create
2. Use the token to clone:
```bash
git clone https://<token>@github.com/microsoft/NOTSOFAR1-Challenge.git
````
**Note**: the token is private, do not share it with anyone!


### Step 2: Install Python 3.10

Python 3.10 is required to run the project. To install it, run the following commands:

```bash
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10
```

### Step 3: Set Up the Python Virtual Environment

Python virtual environments are used to isolate Python dependencies. To set it up, run the following commands:

```bash
sudo apt-get install python3.10-venv
python3.10 -m venv /path/to/virtualenvs/NOTSOFAR
source /path/to/virtualenvs/NOTSOFAR/bin/activate
```

### Step 4: Install Python Dependencies

Navigate to the cloned repository and install the required Python dependencies:

```bash
cd /path/to/NOTSOFAR-Repo
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel Cython fasttext-wheel
sudo apt-get install python3.10-dev ffmpeg build-essential
pip install -r requirements.txt
```

### Step 5: Install Azure CLI

Azure CLI is required to download the datasets. To install it, run the following commands:

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

# Running the inference pipeline
The following command will run the inference pipeline on the dev-set of the recorded meeting dataset.
```bash
cd /path/to/NOTSOFAR-Repo
python run_inference.py
```

The first time you run it, it will automatically download the required models and datasets from blob storage:


1. The development set of the meeting dataset (dev-set) is stored in the `artifacts/meeting_data` directory.
2. The CSS models required to run the inference pipeline are stored in the `artifacts/css_models` directory. 

Outputs will be written to the `artifacts/outputs` directory.

### Running on a subset of the dev-set meeting data
`run_inference.py` by default points to the config yaml that loads the full meeting dataset: 

```
conf_file = project_root / 'configs/inference/css_passthrough_ch0.yaml'
```

For debugging, to run on only one meeting and the Whisper 'tiny' model, you can use the following config:
```
conf_file = project_root / 'configs/inference/css_passthrough_ch0_debug.yaml'
```

The `session_query` argument found in the yaml config file offers more control over filtering meetings.
Note that to submit results on the dev-set, you must evaluate on the full set and no filtering must be performed.

# Running CSS (continuous speech separation) training

## 1. Local training on a data sample for development and debugging
The following command will run CSS training on the 10-second simulated training data sample in `sample_data/css_train_set`.
```bash
cd /path/to/NOTSOFAR-Repo
python run_training_css.local.py
```

## 2. Training on the full simulated training dataset

### Step 1: Download the simulated training dataset
You can use the `download_simulated_subset` function in `utils/azure_storage.py` to download the training dataset from blob storage.
You have the option to download either the complete dataset, comprising almost 1000 hours, or a smaller, 200-hour subset.

For example, to download the entire 1000-hour dataset, make the following calls to download both the training and validation subsets:
```python
train_set_path = download_simulated_subset(
    version='v1.4', volume='1000hrs', subset_name='train', destination_dir=os.path.join(my_dir, 'train'))

train_set_path = download_simulated_subset(
    version='v1.4', volume='1000hrs', subset_name='val', destination_dir=os.path.join(my_dir, 'val'))
```

### Step 2: Run CSS training
Once you have downloaded the training dataset, you can run CSS training on it using the `run_training_css` function in `css/training/train.py`.
The `main` function in `run_training_css.py` provides an entry point with `conf`, `data_root_in`, and `data_root_out` arguments that you can use to configure the run.

It is important to note that the setup and provisioning of a compute cloud environment for running this training process is the responsibility of the user. Our code is designed to support **PyTorch's Distributed Data Parallel (DDP)** framework. This means you can leverage multiple GPUs across several nodes efficiently.




# NOTSOFAR-1 Datasets - Download Instructions
The NOTSOFAR-1 Challenge provides two datasets: a recorded meeting dataset and a simulated training dataset. <br>
This section is for those who are specifically interested in downloading these datasets.
The datasets are stored in Azure Blob Storage, to download them, you will need to install `Azure CLI` ([Project Setup > step 5](###-Step-5:-Install-Azure-CLI)).

You can use either the python utilities in `utils/azure_storage.py` or the `az storage blob download-batch` command to download the datasets as described below.

### Meeting Dataset for Benchmarking and Training

The NOTSOFAR-1 Recorded Meeting Dataset is a collection of 315 meetings, each averaging 6 minutes, recorded across 30 conference rooms with 4-8 attendees, featuring a total of 35 unique speakers. This dataset captures a broad spectrum of real-world acoustic conditions and conversational dynamics.

### Download

To download the dataset, replace the arguments and run the following command:

`--destination` replace with a path to the directory where you want to download the benchmarking dataset (destination directory must exist). <br>
`--pattern` replace the argument with the type of the dataset you want to download (`dev_set` / `eval_set` / `train_set`).

```bash
az storage blob download-batch --destination <path to NOTSOFAR datasets>/benchmark --source https://notsofarsa.blob.core.windows.net/benchmark-datasets --pattern <set type>/*
```

### Simualted Training Dataset

The NOTSOFAR-1 Training Dataset is a 1000-hour simulated training dataset, synthesized with enhanced authenticity for real-world generalization, incorporating 15,000 real acoustic transfer functions.

### Download

To download the dataset, replace the arguments and run the fallowing command:

`--destination` - replace with a path to the directory where you want to download the training dataset (destination directory must exist). <br>
`--source` - replace with the link to the training dataset. <br>
`--pattern` replace with the dataset you want to download:
- `version` - replace with the version of the dataset you want to download (`v1` / `v1.1` / `v1.2` / `v1.3`...)
- `volume` - replace with the volume of the dataset you want to download (`200hrs` / `1000hrs`...)
- `set type` - replace with the type of the dataset you want to download (`dev_set` / `eval_set` / `train_set`).

```bash
az storage blob download-batch --destination <path to NOTSOFAR datasets>/training --source https://notsofarsa.blob.core.windows.net/css-datasets --pattern <version>/<volume>/<set type>/*
```

