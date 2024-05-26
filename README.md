# PTB-XL ECG Classification: A Comparative Study of ML and DL Techniques

## Project Overview
This project aims to classify electrocardiogram (ECG) signals using both traditional machine learning (ML) and advanced deep learning (DL) techniques. It compares the performance of these approaches in diagnosing various cardiac conditions from ECG data.

## Prerequisites
- Python 3.10.13
- Conda
## Directory Structure
- **src/heartDiseaseClassification**: Contains the core ML code.
- **config**: Configuration files for the project.
- **research**: Jupyter notebooks for experimentation.
<!-- - **templates** and **static**: Files for the web interface. -->

## Models Used
1. **Machine Learning Models**:
    - Decision Tree Classifier
    - Random Forest Classifier
    - Logistic Regression

2. **Deep Learning Models**:
    - Simple Convolutional Neural Network (CNN)
    - Long Short-Term Memory (LSTM)
    - Complex CNN (ECG-Classifier)

## Dataset
The project uses the PTB-XL dataset, which contains 12-lead ECG recordings from patients with various cardiac conditions. [Link](https://physionet.org/content/ptb-xl/1.0.3/)

## Results
- The Complex CNN (ECG-Classifier) model achieved the best performance with an 80% classification accuracy and a 90% ROC-AUC.


## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/ukantjadia/Major-Project.git
cd Major-Project
```

### Step 2: Create and Activate Conda Environment
```bash
conda create -n ecg_classification python=3.10.13 -y
conda activate ecg_classification
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
In Conda environment:
```bash
conda env config vars set MY_VARIABLE=my_value
```
In standard Python environment:
```bash
export MY_VARIABLE=my_value
```

### Step 5: Run the Application
```bash
python app.py
```
Open your web browser and navigate to `http://localhost:5000` to access the application.




## License
This project is licensed under the MIT License.

