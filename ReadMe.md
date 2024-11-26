# **Deep Learning Challenge: Alphabet Soup Charity Model**

## **Overview of the Analysis**

The goal of this project was to develop a deep learning model capable of predicting the success of applicants who received funding from Alphabet Soup, a nonprofit organization. By analyzing historical funding data, we built a binary classification model using TensorFlow and Keras to determine whether the funding provided would lead to a successful outcome for each applicant.

---

## **Results**

### **1. Data Preprocessing**

1. **Target Variable:**
   * The target variable for the model is `IS_SUCCESSFUL`, which indicates whether the funding resulted in success.
2. **Feature Variables:**
   * All other columns in the dataset except for `EIN` and `NAME` were considered as feature variables.
3. **Dropped Columns:**
   * `EIN` and `NAME` columns were removed from the dataset as they were not beneficial for the model (neither features nor targets).
4. **Handling Categorical Variables:**
   * For `APPLICATION_TYPE` and `CLASSIFICATION`, rare categories with fewer data points were grouped into a new category called `"Other"`.
   * One-hot encoding was applied to transform categorical variables into numeric format.
5. **Data Scaling:**
   * Feature data (`X`) was scaled using `StandardScaler` to normalize the input values.
6. **Training and Testing Split:**
   * The preprocessed data was split into training (80%) and testing (20%) datasets.

---

### **2. Model Compilation, Training, and Evaluation**

1. **Initial Model Architecture:**
   * **Input Features:** The input layer used all scaled features as its input (`X_train_scaled.shape[1]`).
   * **Hidden Layers:**
     * First hidden layer: 80 neurons, ReLU activation.
     * Second hidden layer: 40 neurons, ReLU activation.
   * **Output Layer:**
     * A single neuron with a sigmoid activation function to output probabilities for binary classification.
2. **Model Compilation:**
   * **Optimizer:** Adam
   * **Loss Function:** Binary cross-entropy
   * **Metrics:** Accuracy
3. **Model Training:**
   * The model was trained for 50 epochs with a batch size of 32.
4. **Initial Model Results:**
   * Loss: **0.7109**
   * Accuracy: **52.81%**

---

### **3. Model Optimization**

Three optimization attempts were made to improve the model’s performance:

1. **Optimization 1: Increased Neurons and Added Layers**
   * Added a third hidden layer with 25 neurons.
   * Increased neurons in the first and second hidden layers to 100 and 50, respectively.
   * Result: Slight improvement in accuracy, but still below the 75% target.
2. **Optimization 2: Added Dropout Layers**
   * Added dropout layers (20% dropout rate) after each hidden layer to prevent overfitting.
   * Result: Improved model stability, but accuracy remained below the target.
3. **Optimization 3: Adjusted Learning Rate and Activation Functions**
   * Changed the activation function in hidden layers to `tanh`.
   * Reduced the learning rate in the Adam optimizer to 0.001.
   * Result: Improved loss values but only marginally increased accuracy.
4. **Optimized Model Results:**
   * Loss: **0.6789**
   * Accuracy: **59.43%**

Despite optimization attempts, the model's accuracy did not reach the target of 75%.

---

## **Summary**

1. **Findings:**
   * The initial model achieved an accuracy of 52.81%, while the optimized model improved marginally to 59.43%.
   * This suggests that the dataset may not have sufficient features or variability to achieve high accuracy with the current approach.
2. **Recommendations:**
   * Consider alternative machine learning algorithms such as:
     * **Random Forests** or **Gradient Boosting:** Tree-based models may better handle categorical variables and relationships in the data.
     * **Support Vector Machines (SVM):** Effective for classification with smaller datasets.
   * Perform feature engineering to extract additional meaningful features.
   * Address class imbalance if the dataset has a skewed distribution of success vs. failure.
3. **Future Work:**
   * Collect additional data points or external datasets to enrich the feature set.
   * Investigate the relationship between input features and the target variable more deeply to improve feature selection.
