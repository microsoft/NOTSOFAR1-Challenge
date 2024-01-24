# Introduction 
Welcome to the Natural Office Talkers in Settings of Far Field Audio Recordings (NOTSOFAR-1) Challenge. This challenge is the first of its kind, focusing on distant speaker diarization and automatic speech recognition (DASR) in far-field meeting scenarios.

Our challenge provides a platform for two new datasets and a baseline system. 

The first dataset is a benchmarking dataset consisting of 315 meetings, each averaging 6 minutes, recorded across 30 conference rooms with 4-8 attendees, featuring a total of 35 unique speakers. This dataset captures a broad spectrum of real-world acoustic conditions and conversational dynamics.

The second dataset is a 1000-hour simulated training dataset, synthesized with enhanced authenticity for real-world generalization, incorporating 15,000 real acoustic transfer functions.

The tasks in this challenge focus on single-device DASR, where multi-channel devices always share the same known geometry. This aligns with common setups in actual conference rooms and avoids technical complexities associated with multi-device tasks, allowing for the development of geometry-specific solutions.

The NOTSOFAR-1 Challenge aims to advance research in the field of distant conversational speech recognition. We provide key resources to unlock the potential of data-driven methods, which we believe are currently constrained by the absence of comprehensive high-quality training and benchmarking datasets. Join us in this exciting journey to push the boundaries of distant conversational speech recognition.


# Project Setup
This fallowing steps will guide you through setting up the project on your machine. <br>
This guide is written for Linux at the moment. Windows support is coming soon.

### Step 1: Clone the Repository

Clone the repository from Azure DevOps. Open your terminal and run the following command:

```bash
sudo apt-get install git
git clone https://dev.azure.com/msatli/atli/_git/NOTSOFAR
```
**Note**: You will be prompted to enter your user credentials. <br>
To generate your credentials, follow these steps:
1. Navigate to the repository on Azure DevOps.
2. Click the **Clone** button.
3. Click the **Generate Git Credentials** button.
4. Enter the generated credentials in the terminal.

### Step 2: Install Python 3.10

Python 3.10 is required to run the project, to install it, run the following commands:

```bash
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10
```

### Step 3: Set Up the Python Virtual Environment

Python virtual environments are used to isolate Python dependencies, to set it up, run the following commands:

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
pip install --upgrade setuptools wheel
sudo apt-get install python3.10-dev, ffmpeg
pip install -r requirements.txt
```

### Step 5: Install Azure CLI

Azure CLI is required to download the datasets, to install it, run the following commands:

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

# NOTSOFAR-1 Datasets
The NOTSOFAR-1 Challenge provides two datasets: a benchmarking dataset and a training dataset. <br>
The datasets are stored in Azure Blob Storage, to download them, you will need to install `Azure CLI` ([Project Setup > step 5](###-Step-5:-Install-Azure-CLI)).

### Benchmarking Dataset

The NOTSOFAR-1 Benchmarking Dataset is a collection of 315 meetings, each averaging 6 minutes, recorded across 30 conference rooms with 4-8 attendees, featuring a total of 35 unique speakers. This dataset captures a broad spectrum of real-world acoustic conditions and conversational dynamics.

### Download

To download the dataset, replace the arguments and run the fallowing command:

`--destination` replace with a path to the directory where you want to download the benchmarking dataset (destination directory must exist). <br>
`--pattern` replace the argument with the type of the dataset you want to download (`dev_set` / `eval_set` / `train_set`).

```bash
az storage blob download-batch --destination <path to NOTSOFAR datasets>/benchmark --source https://notsofarsa.blob.core.windows.net/benchmark-datasets --pattern <set type>/*
```

### Training Dataset

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

