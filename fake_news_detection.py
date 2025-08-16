import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import joblib

# 1. DATA LOADING
fake_path = r"C:\Users\poola\Downloads\Fake News Detection\Fake.csv"
true_path = r"C:\Users\poola\Downloads\Fake News Detection\True.csv"
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)
fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)[['title', 'text', 'label']]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. TEXT CLEANING
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)          # Remove HTML tags
    text = re.sub(r'http\S+', '', text)          # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)         # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces
    return text
def remove_stopwords(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)
df['clean_text'] = df['text'].apply(clean_text).apply(remove_stopwords)

# 3. DATA VISUALIZATION
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="label")
plt.title("Distribution of Fake vs Real News")
plt.xticks([0, 1], ["Fake", "Real"])
plt.savefig("label_distribution.png")
plt.close()

plt.figure(figsize=(10,5))
df['text_len'] = df['clean_text'].apply(len)
sns.histplot(df[df['label']==0]['text_len'], color='red', bins=40, label='Fake', kde=True)
sns.histplot(df[df['label']==1]['text_len'], color='green', bins=40, label='Real', kde=True)
plt.legend()
plt.title("Text Length Distribution by Label")
plt.savefig("textlen_dist.png")
plt.close()

# 4. FEATURE EXTRACTION
vectorizer = TfidfVectorizer(max_features=7000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODEL SELECTION AND TRAINING
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Naive Bayes': MultinomialNB()
}
results = {}
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake","Real"], yticklabels=["Fake","Real"])
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"confusion_matrix_{name.replace(' ','_')}.png")
    plt.close()
    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:,1]
    else:
        y_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0,1], [0,1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.savefig(f"roc_curve_{name.replace(' ','_')}.png")
    plt.close()

# Choose best model (Logistic Regression here)
best_model = models['Logistic Regression']

# 6. FULL TEST SET PREDICTIONS WITH NEAT OUTPUT
y_pred_full = best_model.predict(X_test)
results_df = pd.DataFrame({
    'Original_Text': df.loc[y_test.index, 'text'],
    'Cleaned_Text': df.loc[y_test.index, 'clean_text'],
    'Actual_Label': y_test,
    'Predicted_Label': y_pred_full
})
label_map = {0: 'Fake News', 1: 'Real News'}
results_df['Actual_Label'] = results_df['Actual_Label'].map(label_map)
results_df['Predicted_Label'] = results_df['Predicted_Label'].map(label_map)

print("\nFull predictions on test dataset:\n")
print(results_df.head(20).to_string(index=False))

# Save full prediction results to CSV
results_df.to_csv('fake_news_full_predictions.csv', index=False)
print("\nFull prediction results saved to 'fake_news_full_predictions.csv'")

# 7. CUSTOM PREDICTION FUNCTION
def predict_news(news_text):
    cleaned = clean_text(news_text)
    cleaned = remove_stopwords(cleaned)
    features = vectorizer.transform([cleaned])
    pred = best_model.predict(features)[0]
    return "Fake News" if pred == 0 else "Real News"

# Example usage
sample_news = "Aliens declared peace treaty in surprise UN meeting."
print(f"\nSample prediction: {predict_news(sample_news)}")

# 8. SAVE VECTORIZER AND MODEL FOR FUTURE USE
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(best_model, "fake_news_model.pkl")
print("\nVectorizer and model saved for future use.")