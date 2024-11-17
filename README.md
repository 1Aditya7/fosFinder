## Predicting Factor of Safety (FOS) of Materials in Chassis Design

### 1. Introduction  
This project predicts the Factor of Safety (FOS) for materials to help in choosing the most suitable material for chassis design 
of vehicles. This project using a machine learning approach. The dataset contains material properties, 
which are crucial for engineering and safety applications. The model leverages a Random Forest Regressor to predict FOS based on these properties.

### 2. Dataset Overview  
The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/purushottamnawale/materials), includes the following features:  
- **Su**: Ultimate tensile strength (MPa)  
- **Sy**: Yield strength (MPa)  
- **E**: Modulus of elasticity (MPa)  
- **G**: Shear modulus (MPa)  
- **mu**: Poisson's ratio (unitless)  
- **Ro**: Density (kg/m³)  
- **Material**: The name of the material (categorical)  
- **FOS**: Factor of Safety (target variable, unitless)  

The goal is to predict `FOS` using the numerical properties `Su`, `Sy`, `E`, `G`, `mu`, and `Ro`.

### 3. Methodology  
1. **Preprocessing**:  
   - Addressed missing values in the dataset.  
   - Converted non-numeric entries into numeric format.  
   - Label-encoded the `Material` feature.  
2. **EDA**:  
   - Explored feature relationships with correlation matrices and pair plots.  
3. **Feature Selection**:  
   - Excluded `Material` and `FOS` from feature set (`X`).  
   - Target variable (`y`) was set as `FOS`.  
4. **Modeling**:  
   - Split the dataset into training and testing sets 80% and 20% respectively.  
   - Trained a Random Forest Regressor.  
   - Evaluated the model using metrics such as RMSE, MAE, and \( $R^2$ \) score.

### 4. Exploratory Data Analysis (EDA)  
- **Correlation Matrix**:  
  ![Correlation Matrix](https://github.com/1Aditya7/fosFinder/blob/main/FOSplots/corr.png)  
  This plot highlights the relationships between features and their correlations with `FOS`. <br>
  Inference: There's a strong positive correlation between Su and Sy, suggesting that they tend to move in the same direction.  

- **Pair Plot**:  
  ![Pair Plot](https://github.com/1Aditya7/fosFinder/blob/main/FOSplots/pair.png)  
  This plot shows pairwise relationships and feature distributions.  <br>
  Inference: There are some outliers visible in the plots, especially in the distributions of "G" and "FOS". These outliers might influence the overall relationship between variables.
   - The distributions of "Su" and "Sy" seem to be right-skewed.
   - The distribution of "G" appears to be bimodal.
   - The distribution of "FOS" is heavily right-skewed.

### 5. Approach  
1. Handled missing and categorical data during preprocessing.  
2. Trained a Random Forest Regressor on selected features.  
3. Made predictions for `FOS` and evaluated the model's performance.  
4. Predicted the `FOS` of a new material: `[440, 325, 207000, 79000, 0.3, 7860]`.

### 6. Results  
- Predicted FOS values for the test set:  
  ```python
  [1.22466, 1.2132, 1.63185, 1.53415, 1.53021, 1.85155, 1.92039, ...]
  ```  
- Predicted FOS for a new material:  
  ```python
  Predicted FOS of the new material is: [1.514]
  ```  

### 7. Error Evaluation Metrics  
- **Root Mean Square Error (RMSE)**: 0.1  
- **Mean Absolute Error (MAE)**: 0.04  

### 8. Accuracy Evaluation Metrics  
- **\( $R^2$ \) Score**: 95.98%  

### 9. Limitations and Future Scope  
**Limitations**:  
1. **Data Quality**: Missing values and categorical handling may introduce bias.  
2. **Feature Representation**: The dataset does not include all material properties influencing FOS (e.g., thermal properties).  
3. **Model Interpretability**: Random Forest, as a black-box model, limits explainability of predictions.  
4. **Overfitting Potential**: The high \( $R^2$ \) score may indicate overfitting to the training data.  

**Future Scope**:  
1. **Feature Engineering**: Incorporate additional features like stress-strain curves or material thermal limits.  
2. **Dataset Expansion**: Utilize more comprehensive material datasets for better generalization.  
3. **Model Tuning**: Optimize Random Forest hyperparameters using techniques like grid search.  
4. **Explainable AI**: Use SHAP values or LIME to enhance interpretability.  

### 10. Conclusions  
The project achieved a high predictive accuracy with an \( $R^2$ \) score of 95.98%, low RMSE, and MAE. These results demonstrate the potential of machine learning in material safety applications. However, enhancing data quality, expanding feature sets, and improving model interpretability will be critical for future advancements.  
