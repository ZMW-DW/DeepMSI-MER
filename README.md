# DeepMSI-MER

## Project Overview

The project implements **DeepMSI-MER**, a multi-modal information fusion-based emotion recognition model. The experiments use the **IEMOCAP** and **MELD** datasets, and a **10-fold cross-validation** is employed for model evaluation. After each fold, the best F1 model and its output results are saved.

## Environment Requirements

To successfully run this experiment, you will need the following hardware and software configurations:

### Hardware Requirements
- At least two H20/96GB machines and 300GB of memory are required to run the experiment.

### Software Requirements
- **Linux** operating system.
- **conda** environment management tool.
- Python 3.x.

## Environment Setup

### Install Dependencies

1. Clone the project and navigate to the project directory:
```sh
      git clone https://github.com/ZMW-DW/DeepMSI-MER.git
      cd DeepMSI-MER
```
2. Create and activate a conda environment:
```sh   
      conda create --name your_env_name python=3.x
      conda activate your_env_name
```
3. Install the project dependencies:
```sh
      pip install torch numpy opencv-python pandas sklearn tqdm
```
### Running IEMOCAP Dataset Experiment

1. Navigate to the IEMOCAP folder:
```sh   
      cd IEMOCAP
```   
2. Run the IEMOCAP dataset experiment:
```sh   
      python run_IEMOCAP.py
```
### Running MELD Dataset Experiment

1. Navigate to the MELD folder:
```sh   
      cd MELD
```   
2. Run the MELD dataset experiment:
```sh   
      python run_MELD.py
```
Later, I will upload the preprocessed datasets to a public website. Please ensure that you place the datasets into the data folder and store them according to the dataset name. The specific storage paths and naming conventions can be referenced from the config file in the IEMOCAP folder, or you can adjust them according to your preference.


The download link for the processed IEMOCAP dataset is as follows:

      https://pan.baidu.com/s/1OXYrDnNdxx72vIrSppdZ1w?pwd=4uaa
      
The download link for the processed MELD dataset is as follows:

      https://pan.baidu.com/s/1oJ19xlG7ad0DQjZ9eM4Ovg?pwd=i76d

