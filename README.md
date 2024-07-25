# Potato-Pathology-Detection
Deep Learning Project 2022-2023

### Project Description: Potato Disease Classification Using Deep Learning

**Objective:**
The aim of this project is to build a convolutional neural network (CNN) model to detect and classify different diseases in potato plants from images. The model is trained on a dataset of potato plant images, which are labeled with the corresponding disease category.

**Dataset:**
The dataset used for this project is sourced from the PlantVillage repository, containing images of potato plants affected by various diseases as well as healthy plants. The dataset is divided into training, validation, and testing sets to evaluate the performance of the model.

**Tools and Libraries:**
- TensorFlow and Keras: For building and training the deep learning model.
- Matplotlib: For visualizing the training progress and results.
- NumPy: For numerical operations.
- Google Colab: For running the code and accessing Google Drive for dataset storage.

**Steps Involved:**

1. **Data Preparation:**
   - Load the dataset from Google Drive.
   - Preprocess the images by resizing and rescaling them.
   - Augment the data using random transformations to improve the model's robustness.
   - Split the dataset into training, validation, and testing sets.

2. **Model Architecture:**
   - A sequential model is built using Keras with several convolutional layers, each followed by max-pooling layers.
   - The architecture includes multiple convolutional layers with ReLU activation, max-pooling layers to reduce spatial dimensions, and a fully connected dense layer for classification.
   - The final layer uses the softmax activation function to output probabilities for each class.

3. **Training the Model:**
   - Compile the model using the Adam optimizer and sparse categorical cross-entropy loss function.
   - Train the model for 15 epochs, monitoring the training and validation accuracy and loss.

4. **Evaluation:**
   - Evaluate the model on the test dataset to assess its performance.
   - Plot the training and validation accuracy and loss over the epochs to visualize the training progress.

5. **Prediction:**
   - Use the trained model to make predictions on new images.
   - Display the predicted class along with the actual class and confidence level for test images.

6. **Saving the Model:**
   - Save the trained model for future use and deployment.

**Results:**
- The model's performance is measured in terms of accuracy on the validation and test sets.
- Visualizations of training/validation accuracy and loss curves provide insights into the model's learning process.

**Conclusion:**
This project demonstrates the use of deep learning techniques to effectively classify potato diseases from images, which can be beneficial for early detection and management of crop health in agriculture.

This project involves a comprehensive workflow from data preparation to model training, evaluation, and deployment, showcasing the practical application of deep learning in agricultural technology.
