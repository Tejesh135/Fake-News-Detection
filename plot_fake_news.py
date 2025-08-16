import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords

# 1. DATA LOADING
fake_path = r"C:\\Users\\poola\\Downloads\\Fake News Detection\\Fake.csv"
true_path = r"C:\\Users\\poola\\Downloads\\Fake News Detection\\True.csv"
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
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text).apply(remove_stopwords)

# 3. CLASS DISTRIBUTION BAR CHART
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="label")
plt.title("Distribution of Fake vs Real News")
plt.xticks([0, 1], ["Fake", "Real"])
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("label_distribution.png")
plt.close()

# 4. TEXT LENGTH HISTOGRAM
df['text_len'] = df['clean_text'].apply(len)
plt.figure(figsize=(10,5))
sns.histplot(df[df['label']==0]['text_len'], color='red', bins=40, label='Fake', kde=True)
sns.histplot(df[df['label']==1]['text_len'], color='green', bins=40, label='Real', kde=True)
plt.legend()
plt.title("Text Length Distribution by Label")
plt.xlabel("Text Length (number of characters)")
plt.ylabel("Count")
plt.savefig("textlen_dist.png")
plt.close()

# 5. WORD FREQUENCY BAR CHARTS
def plot_word_freq(df, label, top_n=20):
    texts = df[df['label']==label]['clean_text']
    all_words = ' '.join(texts).split()
    freq = Counter(all_words)
    common_words = freq.most_common(top_n)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(counts), y=list(words), palette='mako')
    labelname = 'Fake News' if label==0 else 'Real News'
    plt.title(f"Top {top_n} Word Frequencies in {labelname}")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(f"word_freq_{labelname.replace(' ', '_').lower()}.png")
    plt.close()

plot_word_freq(df, label=0, top_n=20)
plot_word_freq(df, label=1, top_n=20)