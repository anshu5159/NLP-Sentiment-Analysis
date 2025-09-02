# IMDB Sentiment Analysis
Movie reviews provide valuable insights into audience opinions, but they are written in unstructured text. This project implements a sentiment analysis pipeline using the IMDB movie reviews dataset. The goal is to classify reviews as **positive** or **negative** by applying Natural Language Processing (NLP) techniques and machine learning models. The workflow includes preprocessing text, feature extraction with TF-IDF, model training, evaluation, and visualization of results.

---

## ✅ Objectives
- Perform text preprocessing (cleaning, tokenization, stopword removal, lemmatization).  
- Apply **TF-IDF vectorization** for feature extraction.  
- Train and evaluate **Logistic Regression** and **Linear SVM** models.  
- Compare performance using accuracy, precision, recall, and F1-score.  
- Visualize results with confusion matrices, sentiment distribution, and word clouds.  

---

## 📂 Dataset
- **Source:** IMDB Movie Reviews Dataset (50,000 reviews).  
- **Columns:**
  - `review`: Movie review text.  
  - `sentiment`: Target label (`positive` or `negative`).  

**Preprocessing applied:**  
- Converted text to lowercase.  
- Removed punctuation, numbers, and stopwords.  
- Lemmatized tokens to base form.  
- Mapped sentiment labels to numeric format (`positive=1`, `negative=0`).  

---

## 🛠️ Tasks & Implementation
### **Task 1: Data Preprocessing**
- Cleaned raw reviews (lowercasing, regex cleaning, stopword removal, lemmatization).  
- Verified processed text with sample outputs.  

### **Task 2: Feature Extraction**
- Used **TF-IDF Vectorizer** with:
  - `min_df=5`
  - `max_features=20000`
  - `ngram_range=(1,2)` (unigrams and bigrams).  

### **Task 3: Model Building**
- Split dataset into **80% training** and **20% testing**.  
- Trained:
  - Logistic Regression  
  - Linear Support Vector Machine (LinearSVC)  

### **Task 4: Evaluation**
- Evaluated models using:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Classification Report  
  - Confusion Matrix  

### **Task 5: Visualization**
- Sentiment distribution bar chart.  
- Word clouds for positive and negative reviews.  

---

## 📊 Results
### Logistic Regression:
- Accuracy: **0.8959**  
- Precision: **0.8962**  
- Recall: **0.8959**  
- F1 Score: **0.8959**  

### Linear SVM:
- Accuracy: **0.8944**  
- Precision: **0.8945**  
- Recall: **0.8944**  
- F1 Score: **0.8944**  

**Comparison:**  
- Both models perform almost equally.  
- Logistic Regression slightly outperforms Linear SVM on all metrics.  
- Logistic Regression also trains faster, making it more scalable.  

---

## 📈 Visualizations
- **Bar Chart:** Shows distribution of positive vs negative reviews.  
- **Confusion Matrices:** Highlight misclassifications for both models.  
- **Word Clouds:** Show common words in positive and negative reviews.  

---

## 🔚 Conclusion
This project successfully built an end-to-end **sentiment analysis pipeline** for the IMDB dataset.  

**Key Findings:**  
- Classical ML models like Logistic Regression and SVM achieve ~90% accuracy.  
- Logistic Regression provides slightly better performance and faster training.  
- Visualizations (confusion matrix, word clouds, sentiment distribution) help in understanding dataset balance and feature importance.
