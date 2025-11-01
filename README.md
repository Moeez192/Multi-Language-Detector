# Machine Learning Language Detector

This project is a **language identification model** built in a Jupyter Notebook (`language_detector.ipynb`). It uses a **classic machine learning pipeline** to train a classifier that can predict the language of a given text snippet.

The model is trained on a dataset of text samples from various languages and evaluated for accuracy.

---

## How It Works

The `language_detector.ipynb` notebook implements the following **end-to-end ML pipeline**:

1. **Load Data**  
   Reads the pre-split training and testing data from the `/data` directory.

2. **Text Preprocessing**  
   Cleans the text data by:
   - Removing punctuation, numbers, and stopwords
   - Converting text to lowercase

3. **Feature Extraction**  
   Converts raw text into numerical format using:
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**  
   - or **Count Vectorizer**

4. **Model Training**  
   Trains a classification algorithm such as:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)

   Trained on vectorized `x_train.txt` and labels from `y_train.txt`.

5. **Evaluation**  
   Evaluates performance on the held-out test set (`x_test.txt`, `y_test.txt`) using:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix

6. **Prediction**  
   Includes a function to predict the language of **new, unseen text**.

---

## Dataset

The data is located in the `/data` directory and organized as follows:

| File | Description |
|------|-------------|
| `x_train.txt` | Training text samples |
| `y_train.txt` | Corresponding language labels for training |
| `x_test.txt`  | Testing text samples |
| `y_test.txt`  | Corresponding language labels for testing |
| `labels.csv`  | Maps language codes (e.g., `en`, `fr`) to full names (e.g., `English`, `French`) |
| `README.txt` / `urls.txt` | Metadata and source information about the dataset |

---

## Installation

To run this project, you need **Python 3** and several data science libraries.

### 1. Clone the Repository

```bash
git clone https://github.com/Moeez192/Multi-Language-Detector.git
cd language-detector
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install jupyter notebook pandas scikit-learn
```

