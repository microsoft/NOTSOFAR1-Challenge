[![Slack][slack-badge]][slack-invite]


[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA

# Introduction 
Welcome to the "NOTSOFAR-1: Distant Meeting Transcription with a Single Device" Challenge.

This repo contains the baseline system code and datasets for the NOTSOFAR-1 Challenge.

- For details on the datasets, baseline system, and tasks, please see our 
[NOTSOFAR-1 paper](https://www.isca-archive.org/interspeech_2024/vinnikov24_interspeech.pdf) or visit [CHiME's official challenge website](https://www.chimechallenge.org/current/task2/index).
- Contact us: join the `chime-8-notsofar` channel on the [CHiME Slack](https://join.slack.com/t/chime-fey5388/shared_invite/zt-1oha0gedv-JEUr1mSztR7~iK9AxM4HOA), or open a [GitHub issue](https://github.com/microsoft/NOTSOFAR1-Challenge/issues). 

### 📊 Baseline Results on NOTSOFAR dev-set-1

Values are presented in `tcpWER / tcORC-WER (session count)` format.
<br>
As mentioned in the [official website](https://www.chimechallenge.org/current/task2/index#tracks), 
systems are ranked based on the speaker-attributed 
[tcpWER](https://github.com/fgnt/meeteval/blob/main/doc/tcpwer.md)
, while the speaker-agnostic [tcORC-WER](https://github.com/fgnt/meeteval) serves as a supplementary metric for analysis.
<br>
We include analysis based on a selection of hashtags from our [metadata](https://www.chimechallenge.org/current/task2/data#metadata), providing insights into how different conditions affect system performance.



|                      | Single-Channel        | Multi-Channel         |
|----------------------|-----------------------|-----------------------|
| All Sessions         | **46.8** / 38.5 (177) | **32.4** / 26.7 (106) |
| #NaturalMeeting      | 47.6 / 40.2 (30)      | 32.3 / 26.2 (18)      |
| #DebateOverlaps      | 54.9 / 44.7 (39)      | 38.0 / 31.4 (24)      |
| #TurnsNoOverlap      | 32.4 / 29.7 (10)      | 21.2 / 18.8 (6)       |
| #TransientNoise=high | 51.0 / 43.7 (10)      | 33.6 / 29.1 (5)       |
| #TalkNearWhiteboard  | 55.4 / 43.9 (40)      | 39.9 / 31.2 (22)      |







# Project Setup
The following steps will guide you through setting up the project on your machine. <br>

### Windows Users
This project is compatible with **Linux** environments. Windows users can refer to [Docker](#docker) or 
[Devcontainer](#devcontainer) sections. <br>
Alternatively, install WSL2 by following the [WSL2 Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install), then install Ubuntu 20.04 from the [Microsoft Store](https://www.microsoft.com/en-us/p/ubuntu-2004-lts/9n6svws3rx71?activetab=pivot:overviewtab). <br>

## Cloning the Repository

Clone the `NOTSOFAR1-Challenge` repository from GitHub. Open your terminal and run the following command:

```bash
sudo apt-get install git
cd path/to/your/projects/directory
git clone https://github.com/microsoft/NOTSOFAR1-Challenge.git
```


## Setting up the environment

### Conda

#### Step 1: Install Conda

Conda is a package manager that is used to install Python and other dependencies.<br>
To install Miniconda, which is a minimal version of Conda, run the following commands:

```bash
miniconda_dir="$HOME/miniconda3"
script="Miniconda3-latest-Linux-$(uname -m).sh"
wget --tries=3 "https://repo.anaconda.com/miniconda/${script}"
bash "${script}" -b -p "${miniconda_dir}"
export PATH="${miniconda_dir}/bin:$PATH"
````
*** You may change the `miniconda_dir` variable to install Miniconda in a different directory.


#### Step 2: Create a Conda Environment 

Conda Environments are used to isolate Python dependencies. <br> 
To set it up, run the following commands:

```bash
source "/path/to/conda/dir/etc/profile.d/conda.sh"
conda create --name notsofar python=3.10 -y
conda activate notsofar 
cd /path/to/NOTSOFAR1-Challenge
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel Cython fasttext-wheel
pip install -r requirements.txt
conda install ffmpeg -c conda-forge -y
```

### PIP

#### Step 1: Install Python 3.10

Python 3.10 is required to run the project. To install it, run the following commands:

```bash
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10
```

#### Step 2: Set Up the Python Virtual Environment

Python virtual environments are used to isolate Python dependencies. <br> 
To set it up, run the following commands:

```bash
sudo apt-get install python3.10-venv
python3.10 -m venv /path/to/virtualenvs/NOTSOFAR
source /path/to/virtualenvs/NOTSOFAR/bin/activate
```

#### Step 3: Install Python Dependencies

Navigate to the cloned repository and install the required Python dependencies:

```bash
cd /path/to/NOTSOFAR1-Challenge
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel Cython fasttext-wheel
sudo apt-get install python3.10-dev ffmpeg build-essential
pip install -r requirements.txt
```

### Docker

Refer to the `Dockerfile` in the project's root for dependencies setup. To use Docker, ensure you have Docker installed on your system and configured to use Linux containers.

### Devcontainer
With the provided `devcontainer.json` you can run and work on the project in a [devctonainer](https://containers.dev/) using, for example, the [Dev Containers VSCode Extension](https://code.visualstudio.com/docs/devcontainers/containers).


# Running evaluation - the inference pipeline
The following command will download the **entire dev-set-1** of the recorded meeting dataset and run the inference pipeline
according to selected configuration. The default is configured to `--config-name dev_set_mc_debug` for quick debugging, 
running on a single session with the Whisper 'tiny' model.
```bash
cd /path/to/NOTSOFAR1-Challenge
python run_inference.py
```

To run on all multi-channel or single-channel dev-set sessions, use the following commands respectively:
```bash
python run_inference.py --config-name full_dev_set_mc
python run_inference.py --config-name full_dev_set_sc
```
The first time `run_inference.py` runs, it will automatically download these required models and datasets from blob storage:


1. The development set of the meeting dataset (dev-set) will be stored in the `artifacts/meeting_data` directory.
2. The CSS models required to run the inference pipeline will be stored in the `artifacts/css_models` directory. 

Outputs will be written to the `artifacts/outputs` directory.



The `session_query` argument found in the yaml config file (e.g. `configs/inference/inference_v1.yaml`) offers more control over filtering meetings.
Note that to submit results on the dev-set, you must evaluate on the full set (`full_dev_set_mc` or `full_dev_set_sc`) and no filtering must be performed.


# Integrating your own models 
The inference pipeline is modular, designed for easy research and extension.
Begin by exploring the following components:
- **Continuous Speech Separation (CSS)**: See `css_inference` in `css.py` . We provide a model pre-trained on NOTSOFAR's simulated training dataset, as well as inference and training code. For more information, refer to the [CSS section](#running-css-continuous-speech-separation-training).
- **Automatic Speech Recognition (ASR)**: See `asr_inference` in `asr.py`. The baseline implementation relies on [Whisper](https://github.com/openai/whisper). 
- **Speaker Diarization**: See `diarization_inference` in `diarization.py`. The baseline implementation relies on the [NeMo toolkit](https://github.com/NVIDIA/NeMo).

### Training datasets
For training and fine-tuning your models, NOTSOFAR offers the **simulated training set** and the training portion of the
**recorded meeting dataset**. Refer to the `download_simulated_subset` and `download_meeting_subset` functions in 
[utils/azure_storage.py](https://github.com/microsoft/NOTSOFAR1-Challenge/blob/main/utils/azure_storage.py#L109), 
or the [NOTSOFAR-1 Datasets](#notsofar-1-datasets---download-instructions) section.


# Running CSS (continuous speech separation) training

## 1. Local training on a data sample for development and debugging
The following command will run CSS training on the 10-second simulated training data sample in `sample_data/css_train_set`.
```bash
cd /path/to/NOTSOFAR1-Challenge
python run_training_css_local.py
```

## 2. Training on the full simulated training dataset

### Step 1: Download the simulated training dataset
You can use the `download_simulated_subset` function in 
[utils/azure_storage.py](https://github.com/microsoft/NOTSOFAR1-Challenge/blob/main/utils/azure_storage.py)
to download the training dataset from blob storage.
You have the option to download either the complete dataset, comprising almost 1000 hours, or a smaller, 200-hour subset.

Examples:
```python
ver='v1.5'  # this should point to the lateset and greatest version of the dataset.

# Option 1: Download the training and validation sets of the entire 1000-hour dataset. 
train_set_path = download_simulated_subset(
    version=ver, volume='1000hrs', subset_name='train', destination_dir=os.path.join(my_dir, 'train'))

val_set_path = download_simulated_subset(
    version=ver, volume='1000hrs', subset_name='val', destination_dir=os.path.join(my_dir, 'val'))


# Option 2: Download the training and validation sets of the smaller 200-hour dataset.
train_set_path = download_simulated_subset(
    version=ver, volume='200hrs', subset_name='train', destination_dir=os.path.join(my_dir, 'train'))

val_set_path = download_simulated_subset(
    version=ver, volume='200hrs', subset_name='val', destination_dir=os.path.join(my_dir, 'val'))
```

### Step 2: Run CSS training
Once you have downloaded the training dataset, you can run CSS training on it using the `run_training_css` function in `css/training/train.py`.
The `main` function in `run_training_css.py` provides an entry point with `conf`, `data_root_in`, and `data_root_out` arguments that you can use to configure the run.

It is important to note that the setup and provisioning of a compute cloud environment for running this training process is the responsibility of the user. Our code is designed to support **PyTorch's Distributed Data Parallel (DDP)** framework. This means you can leverage multiple GPUs across several nodes efficiently.

### Step 3: Customizing the CSS model
To add a new CSS model, you need to do the following:
1. Have your model implement the same interface as our baseline CSS model class `ConformerCssWrapper` which is located
in `css/training/conformer_wrapper.py`. Note that in addition to the `forward` method, it must also implement the 
`separate`, `stft`, and `istft` methods. The latter three methods will be used in the inference pipeline and to 
calculate the loss when training.
2. Create a configuration dataclass for your model. Add it as a member of the `TrainCfg` dataclass in 
`css/training/train.py`.
3. Add your model to the `get_model` function in `css/training/train.py`.



# NOTSOFAR-1 Datasets
This section is for those specifically interested in downloading the NOTSOFAR datasets.<br>
The NOTSOFAR-1 Challenge provides two datasets: a recorded meeting dataset and a simulated training dataset. <br>

The datasets are provided for open research. See the [Data License](#data-license) section.
<br>They are hosted in Azure Blob Storage. See download instructions below.
<br>Visis the [Data section on CHiME's website](https://www.chimechallenge.org/current/task2/data) to explore the data further.


## 1. Recorded Meeting Dataset for Benchmarking and Training

The NOTSOFAR-1 Recorded Meeting Dataset is a collection of 237 meetings, each averaging 6 minutes, recorded across 30 conference rooms with 4-8 attendees, featuring a total of 35 unique speakers. This dataset captures a broad spectrum of real-world acoustic conditions and conversational dynamics.

### Download Instructions

To download the dataset, you can call the python function [download_meeting_subset](https://github.com/microsoft/NOTSOFAR1-Challenge/blob/main/utils/azure_storage.py#L109) within `utils/azure_storage.py`.

Alternatively, using [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) CLI, set these arguments and run the following command:

- `subset_name`: name of split to download (`dev_set` / `eval_set` / `train_set`).
- `version`: version to download. 
- `datasets_path` - path to the directory where you want to download the benchmarking dataset (destination directory must exist). <br>

```bash
azcopy copy https://notsofarsa.blob.core.windows.net/benchmark-datasets/<subset_name>/<version>/MTG <datasets_path>/benchmark --recursive
```

### Latest Train, Dev and Eval Subsets

- `240825.1_train`: Corresponds to Train-set-1 and Train-set-2 from the
[challenge datasets](https://www.chimechallenge.org/current/task2/data#meetings-recordings-dataset), 
except faulty white-noise recordings of sc_rockfall_1 have been removed from 3 meetings.
    ```bash
    azcopy copy https://notsofarsa.blob.core.windows.net/benchmark-datasets/train_set/240825.1_train/MTG . --recursive    
    ```

- `240825.1_dev1`: Same as Dev-set-1 from the challenge. Users should be mindful of speakers overlap: there are 12 speakers, 10 of which are in the training set.

    ```bash
    azcopy copy https://notsofarsa.blob.core.windows.net/benchmark-datasets/dev_set/240825.1_dev1/MTG . --recursive
    ```

- `240629.1_eval_small_with_GT`: Identical to the challenge evaluation set, hence enabling direct comparison to challenge results. This relatively smaller evaluation set is designed for resource-constrained research and includes 80 meetings with 2 devices per track (single-channel/multi-channel), totaling 16 hours for each. Ground-truth is available.
    ```bash
    azcopy copy https://notsofarsa.blob.core.windows.net/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG . --recursive
    ```

- `240825.1_eval_full_with_GT`: A larger evaluation set to facilitate further research and increase statistical significane of performance evaluations. 
It includes 129 meetings and a greater variety of devices: 3 multi-channel, and 6-7 single-channel.
    ```bash
    azcopy copy https://notsofarsa.blob.core.windows.net/benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG . --recursive
    ```


### Comparison to the subsets used during the NOTSOFAR-1 Challenge

The dataset currently available for open research have been modified, with the major differences being:
- In addition to the evaluation set used in the challenge (eval-small), a larger evaluation set (eval-full) is made available.
- Due to legal and quality constraints we unfortunately had to remove Dev-set-2. Instead, Dev-set-1 will serve as the development set for the open dataset.
<br> Dev-set-2 is approved for use exclusively as part of the NOTSOFAR-1 Challenge, but not beyond it.
Challenge participants are only allowed to use Dev-set-2 for publications related to systems developed during the Challenge.

  
### Notes on Quality
- Upcoming upgrades: To ensure maximal annotation quality, the training and development subsets are undergoing transcription upgrades. Please stay tuned for updates.
- The Rockfall1 multi-channel device (mc_rockfall_1), suspected of being faulty due to subjective sound quality, is excluded from eval-small and eval-full but remains in the training and development sets. Users may choose whether to use it.


## 2. Simulated Training Dataset

The simulated training dataset consists of almsot 1000 hours simulated with the same microphone-array geometry as the multi-channel devices in the NOTSOFAR meeting dataset.
It was synthesized with enhanced authenticity for real-world generalization, incorporating 15,000 real acoustic transfer functions.

### Download Instructions


To download the dataset, you can call the python function `download_simulated_subset` within `utils/azure_storage.py`.
Alternatively, using [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) CLI, 
set these arguments and run the following command:

- `version`: version of the train data to download ( `1.5` is the latest).
See doc in `download_simulated_subset` function in `utils/azure_storage.py` for available versions.
- `volume` - volume of the train data to download (`200hrs` / `1000hrs`)
- `subset_name`: train data type to download (`train` / `val`)
- `datasets_path` - path to the directory where you want to download the simulated dataset (destination directory must exist). <br>


```bash
azcopy copy https://notsofarsa.blob.core.windows.net/css-datasets/<version>/<volume>/<subset_name> <datasets_path>/benchmark --recursive 
```

Examples:
```bash
azcopy copy https://notsofarsa.blob.core.windows.net/css-datasets/v1.5/200hrs/train . --recursive
azcopy copy https://notsofarsa.blob.core.windows.net/css-datasets/v1.5/1000hrs/train . --recursive
```

## Citation

If you use the NOTSOFAR datasets or code in your research, please cite the following paper:

```bibtex
@inproceedings{vinnikov24_interspeech,
  title     = {NOTSOFAR-1 Challenge: New Datasets, Baseline, and Tasks for Distant Meeting Transcription},
  author    = {Alon Vinnikov and Amir Ivry and Aviv Hurvitz and Igor Abramovski and Sharon Koubi and Ilya Gurvich and Shai Peer and Xiong Xiao and Benjamin Martinez Elizalde and Naoyuki Kanda and Xiaofei Wang and Shalev Shaer and Stav Yagev and Yossi Asher and Sunit Sivasankaran and Yifan Gong and Min Tang and Huaming Wang and Eyal Krupka},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {5003--5007},
  doi       = {10.21437/Interspeech.2024-1788},
  issn      = {2958-1796},
}
```

## Data License
The data provided in this repository is licensed under the 
Creative Commons Attribution 4.0 International License (CC BY 4.0). 
You are free to use, share, and adapt the data as long as appropriate credit is given. 


# 🤝 Contribute

Please refer to our [contributing guide](CONTRIBUTING.md) for more information on how to contribute!

