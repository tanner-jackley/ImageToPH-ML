# ImageToPH-ML: Machine Learning-Based pH Prediction from Images

## Overview

Advisors: Dr. Aziz Fellah and Dr. Mohammed Meziani

Student Developer: Tanner Jackley

This project explores the process of predicting pH values from images using a linear regression machine learning model. Traditional pH measurements rely too much on human estimates or require expensive equipment. This project aims to replace both processes with a ML pipeline that can estimate pH directly from an image.

Pipeline

```
Image -> Image preprocessing -> HSV color values -> ML model -> Predicted pH
```

Tech Stack

- Python
- scikit-learn
- OpenCV
- NumPy / Pandas

## Current Status

✅ Synthetic dataset created

✅ ML pipeline implemented

✅ Baseline regression model trained

✅ Model evaluation working

✅ Image → HSV preprocessing started

✅ Real image dataset integration

⬜ Advanced models (KNN, Decision Tree)

⬜ Model comparison

⬜ End-to-end prediction system


## Implementation

### 1. Synthetic Data Generation

In order to begin development of the model, synthetic data was generated using AI before working with real images.

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
- Predicted vs actual comparison (based off test values)

This ensures the model is learning meaningful relationships. 


### 4. Image Preprocessing

Image pipeline using CSV

- Load image from filepath using cv2.imread
- Convert to HSV values using cv2.cvtColor
- Extract HSV values using cv2.split into Hue, Value, Saturation
- Normilize HSV values

Return example:

```
[0.0838, 0.7565, 0.9753]
```


## Citations

Elsenety, M.M., Mohamed, M.B.I., Sultan, M.E. et al. Facile and highly precise pH-value estimation using common pH paper based on machine learning techniques and supported mobile devices. Sci Rep 12, 22584 (2022). https://doi.org/10.1038/s41598-022-27054-5

Xiao, M., Liu, Z., Xu, N., Jiang, L., Yang, M., & Yi, C. (2020). A Smartphone-Based Sensing System for On-Site Quantitation of Multiple Heavy Metal Ions Using Fluorescent Carbon Nanodots-Based Microarrays. ACS Sensors, 5(3), 870–878. https://doi.org/10.1021/acssensors.0c00219
