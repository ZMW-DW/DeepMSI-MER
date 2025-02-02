# DeepMSI-MER

## Project Overview

The project implements **DeepMSI-MER**, a multi-modal information fusion-based emotion recognition model. The experiments use the **IEMOCAP** and **MELD** datasets, and a **10-fold cross-validation** is employed for model evaluation. After each fold, the best F1 model and its output results are saved.

## Environment Requirements

To successfully run this experiment, you will need the following hardware and software configurations:

### Hardware Requirements
- At least two **H20/96GB** machines are required to run the experiment.

### Software Requirements
- **Linux** operating system.
- **conda** environment management tool.
- Python 3.x.
- Project dependencies: Please install the required packages from the `requirements.txt` file.

## Environment Setup

### Install Dependencies

1. Clone the project and navigate to the project directory:

   git clone https://github.com/ZMW-DW/DeepMSI-MER.git
   cd DeepMSI-MER
   
2. Create and activate a conda environment:
   
   conda create --name your_env_name python=3.x
   conda activate your_env_name

3. Install the project dependencies:

   pip install -r requirements.txt

### Running IEMOCAP Dataset Experiment

1. Navigate to the IEMOCAP folder:
   
   cd IEMOCAP
   
2. Run the IEMOCAP dataset experiment:
   
   python run_IEMOCAP.py

### Running MELD Dataset Experiment

1. Navigate to the MELD folder:
   
   cd MELD
   
2. Run the MELD dataset experiment:
   
   python run_MELD.py
