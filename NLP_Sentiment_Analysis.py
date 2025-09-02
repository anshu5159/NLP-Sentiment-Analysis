#   ASSIGNMENT  -|
#   -------------|
#   NLP Sentiment Analysis  -|
#   -------------------------|


#   Introduction  -|
#   ---------------|
#   This assignment on Sentiment Analysis of the IMDB dataset focuses on applying Natural Language Processing (NLP) and
#   Machine Learning (ML) techniques to classify movie reviews as positive or negative. The project workflow includes:
#    - Data Preprocessing: Cleaning raw text by lowercasing, removing noise, eliminating stopwords, and lemmatizing tokens.
#    - Feature Extraction: Representing text using TF-IDF vectorization.
#    - Model Training: Implementing and training two supervised ML algorithms — Logistic Regression and Support Vector 
#                      Machine (SVM).
#    - Evaluation: Assessing model performance using accuracy, precision, recall, F1-score, classification reports, and
#                  confusion matrices.
#    - Visualization: Exploring data through sentiment distribution plots and word clouds for positive and negative reviews.
#   The goal is to demonstrate how combining NLP preprocessing with traditional ML algorithms can effectively classify
#   sentiment in large-scale text datasets, providing meaningful insights from unstructured movie reviews.



import pandas as pd
            # loading datasets and handling tabular data
import re
            # library for cleaning text
from nltk.tokenize import word_tokenize
            # tokenizes text into individual words
from nltk.corpus import stopwords
            # list of common words like the, is, at
import nltk
nltk.download('wordnet')
            # WordNet dictionary for lemmatization
from nltk.stem import WordNetLemmatizer
            # reduces words to their base form
from sklearn.feature_extraction.text import TfidfVectorizer
            # converts text into numeric TF-IDF vectors
from sklearn.linear_model import LogisticRegression
            # Logistic Regression model
from sklearn.svm import LinearSVC
            # Support Vector Machine (linear kernel)
from sklearn.model_selection import train_test_split
            # splits dataset into training and test sets
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
            # evaluation metrics for model performance
import matplotlib.pyplot as plt
            # plotting
import seaborn as sns
            # visualization


df = pd.read_csv("IMDB-Dataset.csv")
            # loading dataset
df.head()
            # shows first 5 rows
df.columns
            # lists all columns
df.info()
            # shows information about the DataFrame
df['sentiment'].value_counts()
            # counts positive and negative sentiments

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
            # defining stop words and creating instance of lemmatizer

def text_preprocessing(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmatized = [lemma.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)
            # converting text to tokens, removing stopwords, and lemmatizing each token

df['clean'] = df['review'].apply(text_preprocessing)
df[['review','clean','sentiment']].head()
            # processing the reviews and showing first 5 rows

df['label'] = df['sentiment'].map({'positive':1, 'negative':0})
            # label encoding

vectorizer = TfidfVectorizer(min_df=5, max_features=20000, ngram_range=(1,2))
x = vectorizer.fit_transform(df['clean'])
y = df['label'].values
x.shape
            # converting text to TF-IDF vectors while ignoring the rare words that appear less then 5 times
            # limiting the vocab size to 20k, with uni and bi-grams

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

LR = LogisticRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
y_pred
            # Logistic Regression model

print("\nLogistic Regression model Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
            # prints all the evaluation metrics

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
            # Confusion matrix visualization to show how many reviews were correctly/incorrectly classified

svm = LinearSVC()
svm.fit(x_train, y_train)
y_pred2 = svm.predict(x_test)
y_pred2
            # SVM model

print("\nSVM model Results")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2, average='weighted'))
print("Recall:", recall_score(y_test, y_pred2, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred2, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred2))
            # prints all the evaluation metrics

cm = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
            # Confusion matrix visualization to show how many reviews were correctly/incorrectly classified

def summarize(name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred, average='weighted')
    r = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return [name, acc, p, r, f1]
            # function to summarize model performance

summary = pd.DataFrame([summarize("Logistic Regression", y_test, y_pred),
                        summarize("Linear SVM", y_test, y_pred2)],
                        columns=["Model","Accuracy","Precision","Recall","F1"])
            # creating summary DataFrame

summary.sort_values("F1", ascending=False)
#   Metric                | Logistic Regression | SVM (Linear Kernel)
#   ----------------------|---------------------|---------------------
#   Accuracy              | 0.8959              | 0.8944
#   Precision (weighted)  | 0.8962              | 0.8945
#   Recall (weighted)     | 0.8959              | 0.8944
#   F1 Score (weighted)   | 0.8959              | 0.8944
#   Both models perform almost equally.
#   - Logistic Regression edges out slightly in Accuracy, Precision, Recall, and F1.
#   - SVM is also strong but slower to train on large datasets.
#   - For scalability and speed, Logistic Regression may be preferred here.


counts = df['sentiment'].value_counts()
            # variable to hold sentiment counts
counts.plot(kind='bar')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('IMDB Sentiment Distribution')
plt.show()
            # bar plot showing sentiment distribution

#   pip install wordcloud
from wordcloud import WordCloud

pos_text = " ".join(df.loc[df['label']==1, 'clean'])
neg_text = " ".join(df.loc[df['label']==0, 'clean'])
            # creating positive and negative text for word cloud

pos_cloud = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10,5))
plt.imshow(pos_cloud)
plt.axis('off')
plt.title('Positive Reviews WordCloud')
plt.show()
            # positive reviews word cloud

neg_cloud = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
plt.figure(figsize=(10,5))
plt.imshow(neg_cloud)
plt.axis('off')
plt.title('Negative Reviews WordCloud')
plt.show()
            # negative reviews word cloud


#   Conclusion  -|
#   -------------|
#   The assignment successfully implemented a complete sentiment analysis pipeline on the IMDB dataset, from preprocessing
#   raw reviews through feature engineering to model training, evaluation, and visualization.
#   Key Findings:
#    - Both Logistic Regression and SVM achieved strong performance (~89–90% accuracy and F1-score), showing that classical
#      ML models remain highly effective for sentiment classification tasks.
#    - Logistic Regression slightly outperformed SVM in most metrics, while also training faster and being more scalable
#      for large datasets.
#    - Confusion matrices and classification reports revealed balanced performance across positive and negative classes.
#    - Sentiment distribution and word clouds provided useful visualization of dataset balance and common words in reviews.
#   Overall, the project demonstrated how fundamental NLP techniques, combined with robust ML classifiers, can extract
#   valuable sentiment insights from large collections of unstructured text.
