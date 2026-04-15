# ImageToPH-ML: Machine Learning-Based pH Prediction from Images

## Overview

Advisors: Dr. Aziz Fellah and Dr. Mohammed Meziani

Student Developer: Tanner Jackley

This project builds a machine learning pipeline that predicts pH values from images of pH indicators by extracting color features (HSV) and mapping them to continuous pH values using regression models. Traditional pH measurements rely too much on human estimates or require expensive equipment. This project aims to replace both processes with a ML pipeline that can estimate pH directly from an image.

### Pipeline

```
Image -> Image preprocessing -> HSV color values -> ML model -> Predicted pH
```

### Tech Stack

- Python
- scikit-learn
- OpenCV
- NumPy / Pandas


## Goal

Build a real-time system that predicts pH from smartphone images for low-cost, portable chemical analysis.

Input: image of pH strip  

Output: Predicted pH = 10.21


## Current Status

✅ Synthetic dataset created

✅ ML pipeline implemented

✅ Baseline regression model trained

✅ Model evaluation working

✅ Image → HSV preprocessing started

✅ Real image dataset integration

✅ Advanced models (KNN, Decision Tree)

✅ Model comparison

☑️ End-to-end prediction system


## Implementation

### 1. Synthetic Data Generation

Synthetic data was used initially to prototype and validate the machine learning pipeline before integrating real-world image data.

Example row from dataset:
| Hue | Saturation | Value | pH |
| --- | --- | --- | --- |
| 0.212782432111785 | 0.7667570675423553 | 0.7222811080290041 | 5.243561663863074 |

This same structure would later be implemented using real images. This approach allowed for quicker development of the infrastructure for the project. 


### 2. Data Pipeline & Model Training 

Model: Linear Regression

- Input HSV values (synthetic or from image)
- Load dataset from CSV
- Train model (split 80% train, 20% test)
- Output predicted pH


### 3. Model Evaluation

Evaluation metrics:

- Mean Squared Error (MSE)
- Predicted vs actual comparison (based on test values)

This ensures the model is learning meaningful relationships. 


### 4. Image Preprocessing

Image preprocessing pipeline

- Load image from filepath using cv2.imread
- Convert to HSV values using cv2.cvtColor
- Extract HSV values using cv2.split into Hue, Saturation, Value
- Normalize HSV values

Return example:

```
[0.0838, 0.7565, 0.9753]
```


### 5. Model Comparisons

Models from scikit-learn

#### Linear Regression

<img width="500" height="400" alt="LinearRegression" src="https://github.com/user-attachments/assets/6ce025be-85cd-4f05-beb7-290ae4ee2ed6" />

R² Score: 0.9365

MSE: 0.9219

#### K Neighbors Regressor

<img width="500" height="400" alt="KNeighborsRegressor" src="https://github.com/user-attachments/assets/425eda2a-c6a3-438b-8c12-ba05cefe0c50" />

R² Score: 0.9464

MSE: 0.7782

#### Decision Tree Regressor

<img width="500" height="400" alt="DecisionTreeRegressor" src="https://github.com/user-attachments/assets/724bd150-03a7-49d8-80da-05487cee393f" />

R² Score: 0.9571

MSE: 0.6240

#### Random Forest Regressor

<img width="500" height="400" alt="RandomForestRegressor" src="https://github.com/user-attachments/assets/5c95252d-b45b-456c-8739-0d311c3ba7c3" />

R² Score: 0.9786

MSE: 0.3116


| Model | R² | MSE |
|-|-|-|
| Linear Regression | 0.9365 | 0.9219 |
| K Neighbors Regressor | 0.9464 | 0.7782 |
| Decision Tree Regressor | 0.9571 | 0.6240 |
| Random Forest Regressor | 0.9786 | 0.3116 |


### Key Insights

HSV color space provides better separation of pH-related color changes than RGB

Tree-based models (Random Forest) outperform linear models due to nonlinear relationships between color and pH

Model accuracy improves significantly when using real image data vs synthetic data


### How to run

```
git clone https://github.com/tanner-jackley/ImageToPH-ML.git
cd ImageToPH-ML/src
pip install -r requirements.txt
python main.py --image ../data/meziani_images/'2.58.png'
```

## Citations

Elsenety, M.M., Mohamed, M.B.I., Sultan, M.E. et al. Facile and highly precise pH-value estimation using common pH paper based on machine learning techniques and supported mobile devices. Sci Rep 12, 22584 (2022). https://doi.org/10.1038/s41598-022-27054-5

Xiao, M., Liu, Z., Xu, N., Jiang, L., Yang, M., & Yi, C. (2020). A Smartphone-Based Sensing System for On-Site Quantitation of Multiple Heavy Metal Ions Using Fluorescent Carbon Nanodots-Based Microarrays. ACS Sensors, 5(3), 870–878. https://doi.org/10.1021/acssensors.0c00219
