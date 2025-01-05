# Skin Cancer Classification Project

## Overview
This project involves building and training a convolutional neural network (CNN) to classify skin lesions based on an imbalanced dataset. The dataset includes labeled skin lesion images, and the model aims to accurately predict the lesion category.

## Features
- **Data Preprocessing**: Includes techniques such as resizing images, normalization, and handling class imbalances using augmentation.
- **Model Training**: Utilizes CNN architecture with detailed configurations for layers, activation functions, optimizers, and loss functions.
- **Evaluation**: Evaluates the model using metrics like accuracy, precision, recall, F1-score, and confusion matrices.
- **Visualization**: Includes data visualizations such as class distributions, loss/accuracy plots, and confusion matrices for better insights.

## Requirements
### Software and Libraries
- Python 3.8+
- Libraries:
  - TensorFlow / Keras
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - Seaborn

### Hardware
- GPU support is recommended for faster training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kharal786786.git
   cd DsResearchProject
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the Jupyter Notebook file:
   ```bash
   jupyter notebook Final_Code_Skin_Cancer.ipynb
   ```
2. Run the cells sequentially to:
   - Load and preprocess the dataset.
   - Train the CNN model.
   - Evaluate the model and visualize results.

## Dataset
The dataset used for this project is not included in this repository. Please download the dataset from [link to dataset source]. Ensure the dataset is structured correctly before running the notebook.

## Key Sections
- **Data Exploration**: Analyzing the dataset's characteristics and identifying class imbalances.
- **Model Design**: Building and compiling a CNN tailored for skin lesion classification.
- **Model Training**: Training the model while monitoring performance on validation data.
- **Performance Evaluation**: Generating confusion matrices, ROC curves, and other evaluation metrics.

## Results
- Achieved significant accuracy on the classification task.
- Provided insights into class-wise performance and areas for improvement.

## Contributions
Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or issues, please contact Abdullahriazhfd@gmail.com.
