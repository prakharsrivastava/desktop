# Databricks notebook source
# DBTITLE 1,📚 Introduction - NLP Landscape & Project Overview
# MAGIC %md
# MAGIC # 🚀 Complete NLP Guide - From Basics to Transformers
# MAGIC
# MAGIC ## PROJECT CONTEXT: Building a Multi-Task Text Intelligence System
# MAGIC
# MAGIC ### Kya Banana Hai? (What are we building?)
# MAGIC
# MAGIC Hum ek **comprehensive text intelligence system** banayenge jo multiple NLP tasks kar sakta hai:
# MAGIC - 😊😢😡 **Sentiment Analysis** - Text ka emotion detect karna
# MAGIC - 🏷️ **Named Entity Recognition (NER)** - Names, places, organizations identify karna
# MAGIC - ❓ **Question Answering** - Questions ke answers dena
# MAGIC - ✍️ **Text Generation** - New content create karna
# MAGIC - 🔍 **Semantic Search** - Meaning-based search
# MAGIC
# MAGIC ### Journey Overview
# MAGIC
# MAGIC ```
# MAGIC Foundations → Classical ML → Deep Learning → Transformers → Production
# MAGIC    (1-10)        (11-18)        (19-30)         (31-42)       (43-50)
# MAGIC ```
# MAGIC
# MAGIC ### Why NLP? Real-World Applications
# MAGIC
# MAGIC 1. **Customer Support** - Chatbots, ticket classification
# MAGIC 2. **Content Moderation** - Toxic content detection
# MAGIC 3. **Healthcare** - Medical record analysis
# MAGIC 4. **Finance** - News sentiment, document processing
# MAGIC 5. **E-commerce** - Product reviews, recommendations
# MAGIC
# MAGIC ### What You'll Learn
# MAGIC
# MAGIC ✅ Text preprocessing fundamentals
# MAGIC ✅ Classical ML techniques (BoW, TF-IDF)
# MAGIC ✅ Word embeddings (Word2Vec, GloVe)
# MAGIC ✅ Deep learning (RNN, LSTM, GRU)
# MAGIC ✅ Attention mechanisms
# MAGIC ✅ Transformer architecture from scratch
# MAGIC ✅ Modern models (BERT, GPT, T5)
# MAGIC ✅ Production deployment
# MAGIC
# MAGIC Let's start! 🎯

# COMMAND ----------

# DBTITLE 1,🧹 Text Preprocessing - Foundation of NLP
# MAGIC %md
# MAGIC # Cell 2: Text Preprocessing
# MAGIC
# MAGIC ## Why Preprocessing? 
# MAGIC
# MAGIC Raw text bohot messy hota hai:
# MAGIC - Capital/small letters mixed
# MAGIC - Punctuation marks
# MAGIC - Extra spaces
# MAGIC - Special characters
# MAGIC - Numbers
# MAGIC
# MAGIC **Goal**: Text ko clean aur standardized format mein convert karna
# MAGIC
# MAGIC ## Key Techniques
# MAGIC
# MAGIC 1. **Lowercasing** - Sab kuch lowercase mein
# MAGIC 2. **Tokenization** - Text ko words/sentences mein split karna
# MAGIC 3. **Removing Punctuation** - Special characters hatana
# MAGIC 4. **Removing Numbers** - Agar zarurat na ho
# MAGIC 5. **Removing Extra Whitespace** - Extra spaces clean karna

# COMMAND ----------

# DBTITLE 1,Text Preprocessing Implementation
# Install required libraries
%pip install nltk pandas numpy matplotlib seaborn wordcloud scikit-learn -q

import re
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

print("✅ Libraries imported successfully!")

# Sample text for our project
sample_texts = [
    "I absolutely LOVE this product! Best purchase ever!!! 😊",
    "This is terrible... Waste of money. Very disappointed 😞",
    "Apple Inc. announced new iPhone 15 in Cupertino, California.",
    "The service was okay, nothing special. Price: $299.99",
    "AMAZING QUALITY!!! Will buy again. Customer service 10/10!!!"
]

print("\n📝 Sample Texts:")
for i, text in enumerate(sample_texts, 1):
    print(f"{i}. {text}")

# Preprocessing function
def preprocess_text(text, remove_numbers=True, remove_punct=True):
    """
    Text ko clean karne ka complete function
    """
    # Original text
    original = text
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Step 4: Remove numbers (optional)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Step 5: Remove punctuation (optional)
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 6: Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Test preprocessing
print("\n🧹 Preprocessing Results:\n")
for i, text in enumerate(sample_texts, 1):
    cleaned = preprocess_text(text)
    print(f"Original {i}: {text}")
    print(f"Cleaned {i}:  {cleaned}")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,🌿 Stemming & Lemmatization
# MAGIC %md
# MAGIC # Cell 3: Stemming & Lemmatization
# MAGIC
# MAGIC ## Problem: Word Variations
# MAGIC
# MAGIC ```
# MAGIC run, running, runs, ran → Sab ka same meaning hai!
# MAGIC good, better, best → Related words
# MAGIC ```
# MAGIC
# MAGIC ## Solution: Root Form mein convert karo
# MAGIC
# MAGIC ### 1️⃣ Stemming
# MAGIC - **Crude method** - Word ko root form mein cut kar do
# MAGIC - **Fast** but sometimes **incorrect**
# MAGIC - Example: `running → run`, but `university → univers` ❌
# MAGIC
# MAGIC ### 2️⃣ Lemmatization  
# MAGIC - **Smart method** - Proper root word (lemma) nikalo
# MAGIC - **Slower** but **accurate**
# MAGIC - Example: `running → run`, `better → good` ✅
# MAGIC
# MAGIC ## When to Use?
# MAGIC
# MAGIC - **Stemming**: Speed important ho, approximate results okay ho
# MAGIC - **Lemmatization**: Accuracy important ho, meaning preserve karna ho
# MAGIC
# MAGIC ## Project Impact
# MAGIC Stemming/Lemmatization se vocabulary size reduce hoti hai → Model training faster hota hai

# COMMAND ----------

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Test words
test_words = [
    'running', 'runs', 'ran', 'runner',
    'better', 'good', 'best',
    'university', 'universal', 'universe',
    'computing', 'computer', 'computed',
    'loving', 'loved', 'lovely'
]


# COMMAND ----------

stemmer.stem("better")

# COMMAND ----------

 lemmatizer.lemmatize("better",pos='a') 

# COMMAND ----------

# DBTITLE 1,Stemming vs Lemmatization Implementation

print("🌿 Stemming vs Lemmatization Comparison\n")
print(f"{'Original':<15} {'Stemmed':<15} {'Lemmatized':<15}")
print("="*45)

for word in test_words:
    stemmed = stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word, pos='v')  # pos='v' for verb
    print(f"{word:<15} {stemmed:<15} {lemmatized:<15}")

# Apply to our sample text
sample = "The runners are running faster than they ran yesterday. Computing systems computed better results."

print(f"\n\n📝 Original Text:\n{sample}")

# Download punkt_tab for newer NLTK versions
nltk.download('punkt_tab', quiet=True)

# Tokenize
words = word_tokenize(sample.lower())

# Apply stemming
stemmed_words = [stemmer.stem(word) for word in words]
print(f"\n🔪 Stemmed:\n{' '.join(stemmed_words)}")

# Apply lemmatization
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(f"\n🎯 Lemmatized:\n{' '.join(lemmatized_words)}")

# Comparison metrics
print(f"\n📊 Statistics:")
print(f"Original words: {len(set(words))} unique")
print(f"After stemming: {len(set(stemmed_words))} unique")
print(f"After lemmatization: {len(set(lemmatized_words))} unique")
print(f"\n💡 Vocabulary reduced by {len(set(words)) - len(set(stemmed_words))} words with stemming!")

# COMMAND ----------

# DBTITLE 1,🛑 Stopwords & N-grams
# MAGIC %md
# MAGIC # Cell 4: Stopwords & N-grams
# MAGIC
# MAGIC ## 1️⃣ Stopwords
# MAGIC
# MAGIC ### Kya hain ye?
# MAGIC Common words jo har sentence mein hote hain but meaning add nahi karte:
# MAGIC - `the, is, am, are, was, were, a, an, and, or, but...`
# MAGIC
# MAGIC ### Why Remove?
# MAGIC - Reduce noise
# MAGIC - Focus on **meaningful** words
# MAGIC - Faster processing
# MAGIC
# MAGIC ⚠️ **Warning**: Sometimes stopwords important hote hain!
# MAGIC - "not good" vs "good" - VERY DIFFERENT!
# MAGIC
# MAGIC ## 2️⃣ N-grams
# MAGIC
# MAGIC Single words (unigrams) se context miss hota hai:
# MAGIC
# MAGIC ```
# MAGIC Unigrams: ["New", "York", "City"] → Confusing!
# MAGIC Bigrams: ["New York", "York City"] → Better!
# MAGIC Trigrams: ["New York City"] → Perfect! ✅
# MAGIC ```
# MAGIC
# MAGIC ### Types:
# MAGIC - **Unigram** (1 word): `"good"`
# MAGIC - **Bigram** (2 words): `"very good"`  
# MAGIC - **Trigram** (3 words): `"not very good"`
# MAGIC - **N-gram** (n words): Any combination
# MAGIC
# MAGIC ## Project Use
# MAGIC N-grams help capture **phrases** and **context** - essential for sentiment analysis!

# COMMAND ----------

from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt

# Get English stopwords
stop_words = set(stopwords.words('english'))

print("🛑 Sample Stopwords (first 20):")
print(list(stop_words)[:20])
print(f"\nTotal stopwords: {len(stop_words)}")

# Sample text for analysis
text = """The movie was not good at all. The acting was terrible and the plot was boring. 
I would not recommend this movie to anyone. The special effects were okay but that's it."""

print(f"\n📝 Original Text:\n{text}")

# Tokenize and remove stopwords
words = word_tokenize(text.lower())
filtered_words = [w for w in words if w.isalnum() and w not in stop_words]

print(f"\n✂️ After Removing Stopwords:")
print(' '.join(filtered_words))

print(f"\n📊 Impact:")
print(f"Before: {len(words)} words")
print(f"After: {len(filtered_words)} words")
print(f"Reduced by: {len(words) - len(filtered_words)} words ({(1-len(filtered_words)/len(words))*100:.1f}%)")

# N-grams Analysis
print("\n" + "="*60)
print("📚 N-GRAMS ANALYSIS")
print("="*60)

# Generate n-grams
text_for_ngrams = "New York City is the best city in United States of America"
words_ngram = word_tokenize(text_for_ngrams.lower())

# Unigrams
unigrams = list(words_ngram)
print(f"\n1️⃣ Unigrams (1 word):")
print(unigrams[:10])

# Bigrams
bigrams_list = list(ngrams(words_ngram, 2))
print(f"\n2️⃣ Bigrams (2 words):")
for bg in bigrams_list[:8]:
    print(f"  {' '.join(bg)}")

# Trigrams
trigrams_list = list(ngrams(words_ngram, 3))
print(f"\n3️⃣ Trigrams (3 words):")
for tg in trigrams_list[:8]:
    print(f"  {' '.join(tg)}")

# Most common bigrams in movie review
movie_words = [w for w in word_tokenize(text.lower()) if w.isalnum()]
movie_bigrams = list(ngrams(movie_words, 2))
bigram_freq = Counter(movie_bigrams)


# COMMAND ----------

movie_words

# COMMAND ----------

movie_bigrams

# COMMAND ----------

bigram_freq 

# COMMAND ----------

bigram_freq.most_common(3)

# COMMAND ----------


print(f"\n🔥 Most Common Bigrams in Movie Review:")
for bigram, count in bigram_freq.most_common(5):
    print(f"  '{' '.join(bigram)}': {count} times")

# COMMAND ----------

# DBTITLE 1,Stopwords & N-grams Implementation


# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Word frequency with vs without stopwords
all_word_freq = Counter(words).most_common(10)
filtered_word_freq = Counter(filtered_words).most_common(10)

axes[0].barh([w[0] for w in all_word_freq], [w[1] for w in all_word_freq], color='lightcoral', alpha=0.7)
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top Words (With Stopwords)')
axes[0].invert_yaxis()

axes[1].barh([w[0] for w in filtered_word_freq], [w[1] for w in filtered_word_freq], color='lightgreen', alpha=0.7)
axes[1].set_xlabel('Frequency')
axes[1].set_title('Top Words (Without Stopwords)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print("\n✅ Stopwords remove karne se meaningful words ka focus mil gaya!")

# COMMAND ----------

# DBTITLE 1,🎒 Bag of Words (BoW) - First Classifier
# MAGIC %md
# MAGIC # Cell 5: Bag of Words (BoW)
# MAGIC
# MAGIC ## Concept: Words ko Numbers mein Convert karo!
# MAGIC
# MAGIC Machine Learning models text nahi samajhte - **unhe numbers chahiye!**
# MAGIC
# MAGIC ### How BoW Works?
# MAGIC
# MAGIC ```
# MAGIC Documents:
# MAGIC 1. "I love NLP"
# MAGIC 2. "I love programming"
# MAGIC 3. "NLP is amazing"
# MAGIC
# MAGIC Vocabulary: [I, love, NLP, programming, is, amazing]
# MAGIC
# MAGIC BoW Vectors:
# MAGIC Doc 1: [1, 1, 1, 0, 0, 0]  → I(1), love(1), NLP(1)
# MAGIC Doc 2: [1, 1, 0, 1, 0, 0]  → I(1), love(1), programming(1)
# MAGIC Doc 3: [0, 0, 1, 0, 1, 1]  → NLP(1), is(1), amazing(1)
# MAGIC ```
# MAGIC
# MAGIC ## Advantages ✅
# MAGIC - Simple aur fast
# MAGIC - Easy to understand
# MAGIC - Works well for many tasks
# MAGIC
# MAGIC ## Limitations ❌
# MAGIC - **Word order ignore** hota hai: "not good" = "good not"
# MAGIC - **Context loss**: Meaning miss ho sakta hai
# MAGIC - **Sparse vectors**: Bohot saare zeros
# MAGIC
# MAGIC ## Project Application
# MAGIC Hum BoW use karenge apna **first sentiment classifier** banana!

# COMMAND ----------

# DBTITLE 1,BoW Sentiment Classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Create sentiment dataset
sentiment_data = {
    'text': [
        'I love this product amazing quality',
        'Terrible waste of money very disappointed',
        'Absolutely fantastic highly recommend',
        'Worst purchase ever completely useless',
        'Great value for money very satisfied',
        'Poor quality do not buy',
        'Excellent service very happy',
        'Horrible experience never again',
        'Outstanding product exceeded expectations',
        'Awful quality terrible customer service',
        'Best purchase ever love it',
        'Disgusting product extremely unhappy',
        'Wonderful experience highly satisfied',
        'Pathetic quality waste of time',
        'Amazing results very impressed',
        'Disappointing product not worth it',
        'Superb quality great value',
        'Terrible service very upset',
        'Incredible product absolutely love it',
        'Useless product total disappointment'
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative'
    ]
}

df = pd.DataFrame(sentiment_data)
print("📊 Sentiment Dataset:")
print(df.head(10))
print(f"\nDataset size: {len(df)}")
print(f"Positive: {sum(df['sentiment'] == 'positive')}, Negative: {sum(df['sentiment'] == 'negative')}")

# Create Bag of Words
print("\n🎒 Creating Bag of Words...")
vectorizer = CountVectorizer(max_features=50, stop_words='english')

# Fit and transform
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

print(f"\n📐 BoW Matrix Shape: {X.shape}")
print(f"   - {X.shape[0]} documents")
print(f"   - {X.shape[1]} unique words (features)")

# Show vocabulary
print(f"\n📚 Vocabulary (first 20 words):")
print(vectorizer.get_feature_names_out()[:20])

# Show sample vectors
print(f"\n🔢 Sample BoW Vectors:")
for i in range(3):
    print(f"\nDocument {i+1}: '{df['text'][i]}'")
    vector = X[i].toarray()[0]
    non_zero = [(vectorizer.get_feature_names_out()[j], vector[j]) for j in range(len(vector)) if vector[j] > 0]
    print(f"Sentiment: {df['sentiment'][i]}")
    print(f"Non-zero features: {non_zero}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n✂️ Data Split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing: {X_test.shape[0]} samples")

# Train Naive Bayes Classifier
print("\n🤖 Training Naive Bayes Classifier...")
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - BoW Sentiment Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Test with new examples
test_examples = [
    "This product is absolutely amazing and wonderful",
    "Worst experience ever totally disappointed",
    "Good quality but expensive"
]

print("\n🧪 Testing New Examples:")
for example in test_examples:
    vec = vectorizer.transform([example])
    prediction = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0]
    print(f"\nText: '{example}'")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: Negative={proba[0]:.2%}, Positive={proba[1]:.2%}")

print("\n✅ First classifier ready! Simple but effective! 🎉")

# COMMAND ----------

# DBTITLE 1,📊 TF-IDF - Improving the Classifier
# MAGIC %md
# MAGIC # Cell 6: TF-IDF (Term Frequency - Inverse Document Frequency)
# MAGIC
# MAGIC ## Problem with BoW
# MAGIC
# MAGIC BoW mein sab words ko **equal importance** milti hai:
# MAGIC ```
# MAGIC "the" appears 100 times → High count
# MAGIC "excellent" appears 2 times → Low count
# MAGIC ```
# MAGIC
# MAGIC But "excellent" is MORE meaningful than "the"! 🤔
# MAGIC
# MAGIC ## Solution: TF-IDF
# MAGIC
# MAGIC ### Formula:
# MAGIC ```
# MAGIC TF-IDF = TF × IDF
# MAGIC
# MAGIC TF (Term Frequency) = Word ki frequency in document
# MAGIC IDF (Inverse Document Frequency) = log(Total docs / Docs containing word)
# MAGIC ```
# MAGIC
# MAGIC ### Intuition:
# MAGIC - **TF**: Word kitni baar aaya? (How often?)
# MAGIC - **IDF**: Word kitna rare hai? (How rare?)
# MAGIC - **Common words** (the, is, and) → Low IDF → Low score ⬇️
# MAGIC - **Rare meaningful words** (excellent, terrible) → High IDF → High score ⬆️
# MAGIC
# MAGIC ## Example:
# MAGIC
# MAGIC ```
# MAGIC Doc 1: "good good good"  
# MAGIC Doc 2: "bad bad bad"
# MAGIC Doc 3: "good bad"
# MAGIC
# MAGIC "good" appears in 2/3 docs → Medium IDF
# MAGIC "bad" appears in 2/3 docs → Medium IDF
# MAGIC "the" appears in 3/3 docs → Low IDF (if present)
# MAGIC ```
# MAGIC
# MAGIC ## Project Impact
# MAGIC TF-IDF improves classifier by focusing on **meaningful distinguishing words**!

# COMMAND ----------

# DBTITLE 1,TF-IDF Classifier Implementation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Use same sentiment dataset
print("📊 Using previous sentiment dataset...\n")

# Create TF-IDF vectors
print("📊 Creating TF-IDF vectors...")
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')

X_tfidf = tfidf_vectorizer.fit_transform(df['text'])
y = df['sentiment']

print(f"\n📐 TF-IDF Matrix Shape: {X_tfidf.shape}")

# Compare BoW vs TF-IDF for sample document
sample_doc = df['text'][0]
print(f"\n📝 Sample Document: '{sample_doc}'")

bow_vec = vectorizer.transform([sample_doc]).toarray()[0]
tfidf_vec = tfidf_vectorizer.transform([sample_doc]).toarray()[0]

print("\n⚖️ BoW vs TF-IDF Comparison:")
print(f"{'Word':<15} {'BoW Score':<12} {'TF-IDF Score':<12}")
print("="*40)

for i, word in enumerate(tfidf_vectorizer.get_feature_names_out()):
    if tfidf_vec[i] > 0:
        bow_score = bow_vec[i] if i < len(bow_vec) else 0
        print(f"{word:<15} {bow_score:<12.0f} {tfidf_vec[i]:<12.4f}")

# Train-Test Split for TF-IDF
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42
)

# Train Logistic Regression with TF-IDF
print("\n🤖 Training Logistic Regression with TF-IDF...")
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_tfidf, y_train_tfidf)

# Predictions
y_pred_tfidf = lr_clf.predict(X_test_tfidf)

# Evaluation
accuracy_tfidf = accuracy_score(y_test_tfidf, y_pred_tfidf)
print(f"\n🎯 TF-IDF Model Accuracy: {accuracy_tfidf*100:.2f}%")
print(f"   Previous BoW Accuracy: {accuracy*100:.2f}%")
print(f"   Improvement: {(accuracy_tfidf - accuracy)*100:.2f}%")

print("\n📊 TF-IDF Classification Report:")
print(classification_report(y_test_tfidf, y_pred_tfidf))

# Feature Importance Analysis
print("\n🔍 Most Important Words for Classification:\n")

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()
coefficients = lr_clf.coef_[0]

# Sort by absolute coefficient value
top_positive_idx = np.argsort(coefficients)[-10:][::-1]
top_negative_idx = np.argsort(coefficients)[:10]

print("✅ Top Positive Sentiment Words:")
for idx in top_positive_idx:
    print(f"   {feature_names[idx]:<15} (weight: {coefficients[idx]:.4f})")

print("\n❌ Top Negative Sentiment Words:")
for idx in top_negative_idx:
    print(f"   {feature_names[idx]:<15} (weight: {coefficients[idx]:.4f})")

# Confusion Matrix for TF-IDF
cm_tfidf = confusion_matrix(y_test_tfidf, y_pred_tfidf)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# BoW Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[0].set_title(f'BoW Model (Accuracy: {accuracy*100:.1f}%)')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# TF-IDF Confusion Matrix
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1].set_title(f'TF-IDF Model (Accuracy: {accuracy_tfidf*100:.1f}%)')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.show()

# Test with new examples
test_examples_tfidf = [
    "This product is absolutely amazing and wonderful",
    "Worst experience ever totally disappointed",
    "Good quality but expensive"
]

print("\n🧪 Testing New Examples with TF-IDF:")
for example in test_examples_tfidf:
    vec = tfidf_vectorizer.transform([example])
    prediction = lr_clf.predict(vec)[0]
    proba = lr_clf.predict_proba(vec)[0]
    print(f"\nText: '{example}'")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: Negative={proba[0]:.2%}, Positive={proba[1]:.2%}")

print("\n✅ TF-IDF improved our classifier! 🚀")
print("💡 Key Learning: TF-IDF gives importance to meaningful words!")

# COMMAND ----------

# DBTITLE 1,🧠 Word Embeddings Introduction
# MAGIC %md
# MAGIC # Cell 7: Word Embeddings - The Revolution!
# MAGIC
# MAGIC ## Problem with BoW and TF-IDF
# MAGIC
# MAGIC ```
# MAGIC BoW/TF-IDF:
# MAGIC "king" = [0, 0, 1, 0, 0, ...] 
# MAGIC "queen" = [0, 1, 0, 0, 0, ...]
# MAGIC
# MAGIC 😟 No relationship captured!
# MAGIC ```
# MAGIC
# MAGIC Ye vectors:
# MAGIC - **Sparse** (bohot saare zeros)
# MAGIC - **High-dimensional** (vocabulary size)
# MAGIC - **No semantic meaning** (king aur queen related nahi)
# MAGIC
# MAGIC ## Solution: Word Embeddings!
# MAGIC
# MAGIC ### Kya hai Word Embeddings?
# MAGIC
# MAGIC **Dense vectors** jo words ko **continuous space** mein represent karti hain:
# MAGIC
# MAGIC ```
# MAGIC "king" = [0.2, 0.5, -0.1, 0.8, ...] (200-300 dimensions)
# MAGIC "queen" = [0.3, 0.6, -0.2, 0.7, ...]
# MAGIC "man" = [0.1, 0.3, -0.5, 0.4, ...]
# MAGIC "woman" = [0.2, 0.4, -0.6, 0.3, ...]
# MAGIC ```
# MAGIC
# MAGIC ### Magic Property: Semantic Relationships! ✨
# MAGIC
# MAGIC ```
# MAGIC vector("king") - vector("man") + vector("woman") ≈ vector("queen")
# MAGIC
# MAGIC vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")
# MAGIC ```
# MAGIC
# MAGIC ## Types of Word Embeddings
# MAGIC
# MAGIC 1. **Word2Vec** (2013) - Google
# MAGIC    - CBOW (Continuous Bag of Words)
# MAGIC    - Skip-gram
# MAGIC    
# MAGIC 2. **GloVe** (2014) - Stanford
# MAGIC    - Global Vectors
# MAGIC    
# MAGIC 3. **FastText** (2016) - Facebook
# MAGIC    - Subword information
# MAGIC
# MAGIC ## Why Embeddings are Game-Changers?
# MAGIC
# MAGIC ✅ **Semantic meaning** captured
# MAGIC ✅ **Dense representation** (less memory)
# MAGIC ✅ **Transfer learning** possible
# MAGIC ✅ **Better performance** on all NLP tasks
# MAGIC
# MAGIC ## Project Impact
# MAGIC Embeddings will power our deep learning models for sentiment, NER, and generation!

# COMMAND ----------

# DBTITLE 1,🎯 Word2Vec - Training Custom Embeddings
# MAGIC %md
# MAGIC # Cell 8: Word2Vec Implementation
# MAGIC
# MAGIC ## Two Approaches
# MAGIC
# MAGIC ### 1️⃣ CBOW (Continuous Bag of Words)
# MAGIC ```
# MAGIC Context words → Predict center word
# MAGIC Input: ["I", "love", "___", "learning"]
# MAGIC Output: "NLP"
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Skip-gram  
# MAGIC ```
# MAGIC Center word → Predict context words
# MAGIC Input: "NLP"
# MAGIC Output: ["I", "love", "learning"]
# MAGIC ```
# MAGIC
# MAGIC ## Training Process
# MAGIC
# MAGIC 1. Start with random vectors
# MAGIC 2. Slide window over text
# MAGIC 3. Predict context/target words
# MAGIC 4. Update vectors based on errors
# MAGIC 5. Repeat for many epochs
# MAGIC
# MAGIC ## Parameters
# MAGIC
# MAGIC - **vector_size**: Embedding dimension (100-300)
# MAGIC - **window**: Context window size (5-10)
# MAGIC - **min_count**: Ignore rare words (< 5 occurrences)
# MAGIC - **workers**: Parallel processing
# MAGIC - **sg**: 0=CBOW, 1=Skip-gram
# MAGIC
# MAGIC Let's train Word2Vec on a custom corpus!

# COMMAND ----------

# DBTITLE 1,Word2Vec Training & Analysis
# MAGIC %pip install gensim -q
# MAGIC
# MAGIC from gensim.models import Word2Vec
# MAGIC from gensim.models import KeyedVectors
# MAGIC import warnings
# MAGIC warnings.filterwarnings('ignore')
# MAGIC
# MAGIC # Create a larger corpus for training
# MAGIC corpus_sentences = [
# MAGIC     "my name if Ullu",
# MAGIC     "natural language processing is fascinating",
# MAGIC     "machine learning and deep learning are related",
# MAGIC     "neural networks are powerful",
# MAGIC     "transformers revolutionized natural language processing",
# MAGIC     "bert and gpt are transformer models",
# MAGIC     "sentiment analysis uses machine learning",
# MAGIC     "word embeddings capture semantic meaning",
# MAGIC     "python is great for machine learning",
# MAGIC     "deep learning requires large datasets",
# MAGIC     "natural language understanding is challenging",
# MAGIC     "artificial intelligence includes machine learning",
# MAGIC     "neural networks learn from data",
# MAGIC     "computer vision and natural language processing",
# MAGIC     "supervised learning and unsupervised learning",
# MAGIC     "reinforcement learning is different approach",
# MAGIC     "nlp tasks include classification and generation",
# MAGIC     "language models predict next word",
# MAGIC     "attention mechanism improved performance",
# MAGIC     "word vectors represent semantic relationships",
# MAGIC     "contextual embeddings are more powerful",
# MAGIC     "pre-training and fine-tuning strategy",
# MAGIC     "transfer learning saves computation time",
# MAGIC     "tokenization splits text into words",
# MAGIC     "sentiment can be positive negative or neutral",
# MAGIC     "named entity recognition extracts entities",
# MAGIC     "machine translation converts languages",
# MAGIC     "question answering requires understanding",
# MAGIC     "text summarization condenses information",
# MAGIC     "chatbots use natural language processing",
# MAGIC     "data preprocessing is crucial step"
# MAGIC ]
# MAGIC
# MAGIC print("📚 Training Corpus:")
# MAGIC print(f"Total sentences: {len(corpus_sentences)}")
# MAGIC print(f"Sample sentences:")
# MAGIC for sent in corpus_sentences[:5]:
# MAGIC     print(f"  - {sent}")
# MAGIC
# MAGIC # Tokenize sentences
# MAGIC tokenized_corpus = [sent.split() for sent in corpus_sentences]
# MAGIC
# MAGIC print(f"\nTotal tokens: {sum(len(sent) for sent in tokenized_corpus)}")
# MAGIC print(f"Vocabulary size: {len(set([word for sent in tokenized_corpus for word in sent]))}")
# MAGIC
# MAGIC # Train Word2Vec model (Skip-gram)
# MAGIC print("\n🤖 Training Word2Vec (Skip-gram)...")
# MAGIC model_sg = Word2Vec(
# MAGIC     sentences=tokenized_corpus,
# MAGIC     vector_size=100,        # Embedding dimension
# MAGIC     window=5,               # Context window
# MAGIC     min_count=1,           # Minimum word frequency
# MAGIC     workers=4,             # Parallel processing
# MAGIC     sg=1,                  # Skip-gram (1) vs CBOW (0)
# MAGIC     epochs=100,            # Training iterations
# MAGIC     seed=42
# MAGIC )
# MAGIC
# MAGIC print("✅ Model trained!")
# MAGIC print(f"\n📐 Model Info:")
# MAGIC print(f"   Vocabulary size: {len(model_sg.wv)}")
# MAGIC print(f"   Vector dimensions: {model_sg.wv.vector_size}")
# MAGIC print(f"   Training algorithm: Skip-gram")
# MAGIC
# MAGIC # Get word vector
# MAGIC word = "learning"
# MAGIC vector = model_sg.wv[word]
# MAGIC print(f"\n🔢 Vector for '{word}':")
# MAGIC print(f"   Shape: {vector.shape}")
# MAGIC print(f"   First 10 dimensions: {vector[:10]}")
# MAGIC
# MAGIC # Find similar words
# MAGIC print("\n🔍 Finding Similar Words:\n")
# MAGIC
# MAGIC test_words = ['learning', 'natural', 'neural', 'language',"Animal"]
# MAGIC
# MAGIC for word in test_words:
# MAGIC     if word in model_sg.wv:
# MAGIC         similar = model_sg.wv.most_similar(word, topn=5)
# MAGIC         print(f"✅ Most similar to '{word}':")
# MAGIC         for similar_word, similarity in similar:
# MAGIC             print(f"   {similar_word:<20} (similarity: {similarity:.4f})")
# MAGIC         print()
# MAGIC
# MAGIC # Word analogies (King - Man + Woman = Queen)
# MAGIC print("\n🧠 Testing Word Analogies:\n")
# MAGIC
# MAGIC try:
# MAGIC     # machine + processing - learning = ?
# MAGIC     result = model_sg.wv.most_similar(
# MAGIC         positive=['natural', 'processing'],
# MAGIC         negative=['machine'],
# MAGIC         topn=3
# MAGIC     )
# MAGIC     print("❓ natural + processing - machine = ?")
# MAGIC     for word, score in result:
# MAGIC         print(f"   → {word} (score: {score:.4f})")
# MAGIC except:
# MAGIC     print("   Not enough training data for this analogy")
# MAGIC
# MAGIC print("\n" + "="*60)
# MAGIC
# MAGIC # Train CBOW model for comparison
# MAGIC print("\n🤖 Training Word2Vec (CBOW)...")
# MAGIC model_cbow = Word2Vec(
# MAGIC     sentences=tokenized_corpus,
# MAGIC     vector_size=100,
# MAGIC     window=5,
# MAGIC     min_count=1,
# MAGIC     workers=4,
# MAGIC     sg=0,                  # CBOW
# MAGIC     epochs=100,
# MAGIC     seed=42
# MAGIC )
# MAGIC
# MAGIC print("✅ CBOW model trained!")
# MAGIC
# MAGIC # Compare similarity scores
# MAGIC test_word = "learning"
# MAGIC if test_word in model_sg.wv and test_word in model_cbow.wv:
# MAGIC     print(f"\n⚖️ Skip-gram vs CBOW Comparison for '{test_word}':\n")
# MAGIC     
# MAGIC     sg_similar = model_sg.wv.most_similar(test_word, topn=5)
# MAGIC     cbow_similar = model_cbow.wv.most_similar(test_word, topn=5)
# MAGIC     
# MAGIC     print(f"{'Skip-gram':<30} {'CBOW':<30}")
# MAGIC     print("="*60)
# MAGIC     
# MAGIC     for i in range(5):
# MAGIC         sg_word, sg_score = sg_similar[i]
# MAGIC         cbow_word, cbow_score = cbow_similar[i]
# MAGIC         print(f"{sg_word:<20} ({sg_score:.3f})  {cbow_word:<20} ({cbow_score:.3f})")
# MAGIC
# MAGIC print("\n💡 Key Insights:")
# MAGIC print("   - Skip-gram: Better for rare words, captures rare relationships")
# MAGIC print("   - CBOW: Faster training, good for common words")
# MAGIC print("   - Both capture semantic relationships!")
# MAGIC
# MAGIC # Vocabulary analysis
# MAGIC vocab_words = list(model_sg.wv.index_to_key)
# MAGIC print(f"\n📚 Full Vocabulary ({len(vocab_words)} words):")
# MAGIC print(vocab_words[:20], "...")
# MAGIC
# MAGIC print("\n✅ Word2Vec embeddings ready for use! 🎉")

# COMMAND ----------

# DBTITLE 1,🌍 GloVe & FastText Comparison
# MAGIC %md
# MAGIC # Cell 9: GloVe & FastText
# MAGIC
# MAGIC ## Word2Vec vs GloVe vs FastText
# MAGIC
# MAGIC ### 🎯 Word2Vec (Google, 2013)
# MAGIC **Approach**: Predictive (local context)
# MAGIC - CBOW / Skip-gram
# MAGIC - Learns from nearby words
# MAGIC - Fast training
# MAGIC
# MAGIC **Limitation**: Out-of-vocabulary (OOV) words? ❌
# MAGIC
# MAGIC ### 🌎 GloVe (Stanford, 2014)
# MAGIC **Approach**: Count-based (global statistics)
# MAGIC - Global co-occurrence matrix
# MAGIC - Combines local + global context
# MAGIC - Often better on analogies
# MAGIC
# MAGIC **Limitation**: Still OOV problem ❌
# MAGIC
# MAGIC ### ⚡ FastText (Facebook, 2016)
# MAGIC **Approach**: Subword information
# MAGIC - Breaks words into character n-grams
# MAGIC - "learning" = ["le", "lea", "ear", "arn", "rni", "nin", "ing", "ng"]
# MAGIC - Can handle unknown words! ✅
# MAGIC
# MAGIC **Example**:
# MAGIC ```
# MAGIC Word2Vec/GloVe:
# MAGIC "learning" → one vector
# MAGIC "unlearning" → UNKNOWN ❌
# MAGIC
# MAGIC FastText:
# MAGIC "learning" → sum of subword vectors
# MAGIC "unlearning" → can generate from subwords! ✅
# MAGIC ```
# MAGIC
# MAGIC ## When to Use What?
# MAGIC
# MAGIC | Model | Best For | Speed |
# MAGIC |-------|----------|-------|
# MAGIC | Word2Vec | Large corpus, common words | Fast ⚡ |
# MAGIC | GloVe | Analogies, small corpus | Medium 🐌 |
# MAGIC | FastText | Rare words, morphology, typos | Fast ⚡ |
# MAGIC
# MAGIC ## Pre-trained Embeddings
# MAGIC
# MAGIC Training from scratch is expensive! Use pre-trained:
# MAGIC - **Word2Vec**: Google News (3M vocab, 300d)
# MAGIC - **GloVe**: Wikipedia + Gigaword (400K vocab, 50-300d)  
# MAGIC - **FastText**: Common Crawl (2M vocab, 300d)
# MAGIC
# MAGIC ## Project Application
# MAGIC We'll use pre-trained GloVe for our sentiment and NER tasks!

# COMMAND ----------

# DBTITLE 1,FastText Implementation
# For this demo, we'll show how to use pre-trained embeddings
# In practice, you'd download GloVe/FastText files

from gensim.models.fasttext import FastText
import numpy as np
from scipy.spatial.distance import cosine

print("⚡ FastText Training (with subword info)...\n")

# Train FastText model
model_fasttext = FastText(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,              # Skip-gram
    epochs=100,
    min_n=2,           # Min character n-gram length
    max_n=5,           # Max character n-gram length
    seed=42
)

print("✅ FastText model trained!\n")

# Key difference: FastText can handle OOV words!
print("🧪 Testing Out-of-Vocabulary (OOV) Words:\n")

# Words in vocabulary
in_vocab_word = "learning"
# Similar word but NOT in training corpus
oov_word = "learnings"  # Plural form

print(f"Is '{in_vocab_word}' in vocabulary? {in_vocab_word in model_fasttext.wv}")
print(f"Is '{oov_word}' in vocabulary? {oov_word in model_fasttext.wv}")

# Word2Vec would fail on OOV, but FastText can handle it!
try:
    # FastText can generate vector for OOV words using subword info
    vec_in_vocab = model_fasttext.wv[in_vocab_word]
    vec_oov = model_fasttext.wv[oov_word]  # This works with FastText!
    
    print(f"\n✅ FastText generated vector for OOV word '{oov_word}'!")
    
    # Calculate similarity
    similarity = 1 - cosine(vec_in_vocab, vec_oov)
    print(f"   Similarity between '{in_vocab_word}' and '{oov_word}': {similarity:.4f}")
    print(f"   💡 High similarity because they share subword patterns!")
except Exception as e:
    print(f"   Error: {e}")

# Demonstrate subword breakdown
print(f"\n🔍 Subword N-grams for 'learning':")
# FastText breaks words into character n-grams
word = "learning"
subwords = []
for n in range(3, 6):  # 3-gram to 5-gram
    for i in range(len(word) - n + 1):
        subwords.append(word[i:i+n])

print(f"   {subwords}")
print(f"   💡 FastText learns vectors for these subwords!")

# Compare all three models
print("\n" + "="*70)
print("⚖️ Model Comparison: Word2Vec (Skip-gram) vs CBOW vs FastText")
print("="*70)

test_word = "machine"

if test_word in model_sg.wv and test_word in model_cbow.wv and test_word in model_fasttext.wv:
    print(f"\nMost similar words to '{test_word}':\n")
    
    sg_sim = model_sg.wv.most_similar(test_word, topn=5)
    cbow_sim = model_cbow.wv.most_similar(test_word, topn=5)
    ft_sim = model_fasttext.wv.most_similar(test_word, topn=5)
    
    print(f"{'Skip-gram':<25} {'CBOW':<25} {'FastText':<25}")
    print("-"*75)
    
    for i in range(5):
        sg_w, sg_s = sg_sim[i] if i < len(sg_sim) else ("-", 0)
        cbow_w, cbow_s = cbow_sim[i] if i < len(cbow_sim) else ("-", 0)
        ft_w, ft_s = ft_sim[i] if i < len(ft_sim) else ("-", 0)
        
        print(f"{sg_w:<20}({sg_s:.3f}) {cbow_w:<20}({cbow_s:.3f}) {ft_w:<20}({ft_s:.3f})")

# Demonstrate handling of typos
print("\n" + "="*70)
print("⚡ FastText's Superpower: Handling Typos & Unknown Words")
print("="*70)

typos_and_variations = [
    ("machine", "machne"),      # typo
    ("learning", "lerning"),    # typo
    ("natural", "naturel"),     # typo
]

print("\n🔍 Testing typo tolerance:\n")

for correct, typo in typos_and_variations:
    if correct in model_fasttext.wv:
        try:
            vec_correct = model_fasttext.wv[correct]
            vec_typo = model_fasttext.wv[typo]  # Can generate even for typos!
            
            similarity = 1 - cosine(vec_correct, vec_typo)
            
            print(f"'{correct}' vs '{typo}':")
            print(f"   Similarity: {similarity:.4f}")
            print(f"   {'High! ✅' if similarity > 0.7 else 'Low ❌'}")
            print()
        except:
            pass

# Summary table
print("\n📊 Summary: When to Use Each Model\n")

comparison_data = {
    'Feature': ['Training Speed', 'Memory Usage', 'OOV Handling', 'Rare Words', 'Analogies', 'Subword Info'],
    'Word2Vec': ['Fast ⚡', 'Low ✅', 'No ❌', 'Poor ❌', 'Good ✅', 'No ❌'],
    'GloVe': ['Medium 🐌', 'High ⚠️', 'No ❌', 'Good ✅', 'Best 🏆', 'No ❌'],
    'FastText': ['Fast ⚡', 'Medium 🐌', 'Yes ✅', 'Best 🏆', 'Good ✅', 'Yes ✅']
}

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

print("\n🎯 Recommendation:")
print("   - General use: FastText (best balance)")
print("   - Word analogies: GloVe")
print("   - Speed priority: Word2Vec")
print("   - Morphologically rich languages: FastText")

print("\n✅ Understanding of different embedding methods complete! 🎉")

# COMMAND ----------

# DBTITLE 1,🖼️ Embedding Visualization with t-SNE
# MAGIC %md
# MAGIC # Cell 10: Embedding Visualization
# MAGIC
# MAGIC ## Challenge: Visualizing High-Dimensional Data
# MAGIC
# MAGIC Word embeddings are typically **100-300 dimensions**, but we can only visualize **2D or 3D**!
# MAGIC
# MAGIC ```
# MAGIC 100D embedding → [0.2, 0.5, -0.1, 0.8, ...] ❌ Can't plot!
# MAGIC 2D projection → [0.3, 0.7] ✅ Can plot!
# MAGIC ```
# MAGIC
# MAGIC ## Solution: Dimensionality Reduction
# MAGIC
# MAGIC ### 🎯 t-SNE (t-Distributed Stochastic Neighbor Embedding)
# MAGIC
# MAGIC **What it does**: 
# MAGIC Reduces high-dimensional data to 2D/3D while **preserving local structure**
# MAGIC
# MAGIC **Key Properties**:
# MAGIC - Similar words stay close together
# MAGIC - Dissimilar words move apart
# MAGIC - Forms meaningful clusters
# MAGIC
# MAGIC **How it works**:
# MAGIC 1. Calculate pairwise similarities in high-D space
# MAGIC 2. Initialize random 2D points
# MAGIC 3. Move points to match high-D similarities
# MAGIC 4. Iterate until convergence
# MAGIC
# MAGIC ### Parameters:
# MAGIC - **perplexity**: Balance local vs global structure (5-50)
# MAGIC - **n_iter**: Number of optimization iterations (1000+)
# MAGIC - **learning_rate**: Step size (10-1000)
# MAGIC
# MAGIC ## Alternative: PCA
# MAGIC - **PCA**: Faster, linear, preserves global structure
# MAGIC - **t-SNE**: Slower, non-linear, better clusters
# MAGIC
# MAGIC ## Project Use
# MAGIC Visualization helps us:
# MAGIC - Understand word relationships
# MAGIC - Debug embedding quality
# MAGIC - Discover semantic clusters
# MAGIC - Present findings to stakeholders

# COMMAND ----------

# DBTITLE 1,t-SNE Visualization Implementation
# MAGIC %pip install scikit-learn plotly -q
# MAGIC
# MAGIC from sklearn.manifold import TSNE
# MAGIC from sklearn.decomposition import PCA
# MAGIC import plotly.graph_objects as go
# MAGIC import plotly.express as px
# MAGIC
# MAGIC print("🖼️ Visualizing Word Embeddings with t-SNE\n")
# MAGIC
# MAGIC # Get word vectors from our trained model
# MAGIC vocab_words = list(model_fasttext.wv.index_to_key[:50])  # Top 50 words
# MAGIC vectors = np.array([model_fasttext.wv[word] for word in vocab_words])
# MAGIC
# MAGIC print(f"📐 Data Shape: {vectors.shape}")
# MAGIC print(f"   Words: {len(vocab_words)}")
# MAGIC print(f"   Dimensions: {vectors.shape[1]}")
# MAGIC
# MAGIC # Apply t-SNE
# MAGIC print("\n🔄 Applying t-SNE dimensionality reduction...")
# MAGIC tsne = TSNE(
# MAGIC     n_components=2,
# MAGIC     perplexity=15,
# MAGIC     n_iter=1000,
# MAGIC     random_state=42,
# MAGIC     verbose=0
# MAGIC )
# MAGIC
# MAGIC embeddings_2d = tsne.fit_transform(vectors)
# MAGIC print("✅ t-SNE complete!")
# MAGIC print(f"   Reduced to: {embeddings_2d.shape}")
# MAGIC
# MAGIC # Create interactive plot with Plotly
# MAGIC fig = go.Figure()
# MAGIC
# MAGIC # Add scatter plot
# MAGIC fig.add_trace(go.Scatter(
# MAGIC     x=embeddings_2d[:, 0],
# MAGIC     y=embeddings_2d[:, 1],
# MAGIC     mode='markers+text',
# MAGIC     text=vocab_words,
# MAGIC     textposition='top center',
# MAGIC     marker=dict(
# MAGIC         size=10,
# MAGIC         color=np.arange(len(vocab_words)),
# MAGIC         colorscale='Viridis',
# MAGIC         showscale=True,
# MAGIC         colorbar=dict(title="Word Index")
# MAGIC     ),
# MAGIC     textfont=dict(size=9),
# MAGIC     hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
# MAGIC ))
# MAGIC
# MAGIC fig.update_layout(
# MAGIC     title='t-SNE Visualization of Word Embeddings (FastText)',
# MAGIC     xaxis_title='t-SNE Dimension 1',
# MAGIC     yaxis_title='t-SNE Dimension 2',
# MAGIC     width=900,
# MAGIC     height=700,
# MAGIC     hovermode='closest'
# MAGIC )
# MAGIC
# MAGIC fig.show()
# MAGIC
# MAGIC print("\n🔍 Observations:")
# MAGIC print("   - Words with similar meanings cluster together")
# MAGIC print("   - 'learning', 'machine', 'deep' form ML cluster")
# MAGIC print("   - 'natural', 'language', 'processing' form NLP cluster")
# MAGIC
# MAGIC # Compare t-SNE vs PCA
# MAGIC print("\n" + "="*60)
# MAGIC print("⚖️ Comparing t-SNE vs PCA")
# MAGIC print("="*60)
# MAGIC
# MAGIC # Apply PCA
# MAGIC pca = PCA(n_components=2, random_state=42)
# MAGIC embeddings_pca = pca.fit_transform(vectors)
# MAGIC
# MAGIC print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
# MAGIC print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
# MAGIC
# MAGIC # Create side-by-side comparison
# MAGIC fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# MAGIC
# MAGIC # t-SNE plot
# MAGIC ax1 = plt.subplot(1, 2, 1)
# MAGIC scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
# MAGIC                       c=np.arange(len(vocab_words)), cmap='viridis', alpha=0.6, s=100)
# MAGIC for i, word in enumerate(vocab_words):
# MAGIC     ax1.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
# MAGIC                 fontsize=8, alpha=0.8)
# MAGIC ax1.set_title('t-SNE (Non-linear, Local Structure)', fontsize=12, fontweight='bold')
# MAGIC ax1.set_xlabel('Component 1')
# MAGIC ax1.set_ylabel('Component 2')
# MAGIC ax1.grid(True, alpha=0.3)
# MAGIC
# MAGIC # PCA plot
# MAGIC ax2 = plt.subplot(1, 2, 2)
# MAGIC scatter2 = ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
# MAGIC                       c=np.arange(len(vocab_words)), cmap='viridis', alpha=0.6, s=100)
# MAGIC for i, word in enumerate(vocab_words):
# MAGIC     ax2.annotate(word, (embeddings_pca[i, 0], embeddings_pca[i, 1]), 
# MAGIC                 fontsize=8, alpha=0.8)
# MAGIC ax2.set_title('PCA (Linear, Global Structure)', fontsize=12, fontweight='bold')
# MAGIC ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
# MAGIC ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
# MAGIC ax2.grid(True, alpha=0.3)
# MAGIC
# MAGIC plt.colorbar(scatter1, ax=[ax1, ax2], label='Word Index')
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC print("\n💡 Key Differences:")
# MAGIC print("   t-SNE:")
# MAGIC print("      - Better clusters and local relationships")
# MAGIC print("      - Good for exploration and visualization")
# MAGIC print("      - Slower, random (different runs = different plots)")
# MAGIC print("\n   PCA:")
# MAGIC print("      - Preserves global variance structure")
# MAGIC print("      - Deterministic (same every time)")
# MAGIC print("      - Faster, good for preprocessing")
# MAGIC
# MAGIC print("\n✅ Embedding visualization complete! 🎉")
# MAGIC print("💡 Part 1 (NLP Foundations) finished! Ready for Classical ML! 🚀")

# COMMAND ----------

# DBTITLE 1,🎯 PART 2: Traditional ML for NLP
# MAGIC %md
# MAGIC # 🎓 PART 2: Traditional ML for NLP
# MAGIC
# MAGIC ## What We'll Build
# MAGIC
# MAGIC In this section, we'll use traditional ML algorithms with text features to solve real NLP tasks:
# MAGIC
# MAGIC ### 🏆 Key Tasks:
# MAGIC 1. **Text Classification** - Categorize documents
# MAGIC 2. **Named Entity Recognition (NER)** - Extract entities
# MAGIC 3. **Part-of-Speech (POS) Tagging** - Label word roles
# MAGIC 4. **Sentiment Analysis** - End-to-end pipeline
# MAGIC
# MAGIC ### 🛠️ Techniques:
# MAGIC - Naive Bayes
# MAGIC - Logistic Regression
# MAGIC - Support Vector Machines (SVM)
# MAGIC - Decision Trees / Random Forests
# MAGIC - Feature Engineering
# MAGIC
# MAGIC ### 📊 Evaluation:
# MAGIC - Accuracy, Precision, Recall, F1-Score
# MAGIC - Cross-validation strategies
# MAGIC - Handling imbalanced data
# MAGIC
# MAGIC Let's start building! 🚀

# COMMAND ----------

# DBTITLE 1,📄 Text Classification - ML Approaches
# MAGIC %md
# MAGIC # Cell 11: Text Classification
# MAGIC
# MAGIC ## Task: Document ko categories mein classify karna
# MAGIC
# MAGIC ### Real-World Examples:
# MAGIC - **Spam Detection**: Email spam hai ya ham?
# MAGIC - **News Categorization**: Sports, Politics, Tech, Entertainment?
# MAGIC - **Support Ticket Routing**: Bug, Feature Request, Question?
# MAGIC
# MAGIC ## Pipeline:
# MAGIC
# MAGIC ```
# MAGIC Raw Text → Preprocessing → Feature Extraction → ML Model → Prediction
# MAGIC ```
# MAGIC
# MAGIC ## Popular Algorithms:
# MAGIC
# MAGIC ### 1️⃣ Naive Bayes
# MAGIC - **Assumption**: Features are independent
# MAGIC - **Pros**: Fast, works well with small data
# MAGIC - **Best for**: Text classification, spam detection
# MAGIC
# MAGIC ### 2️⃣ Logistic Regression
# MAGIC - **Approach**: Linear model with sigmoid
# MAGIC - **Pros**: Interpretable, probability scores
# MAGIC - **Best for**: Binary/multi-class classification
# MAGIC
# MAGIC ### 3️⃣ Support Vector Machines (SVM)
# MAGIC - **Approach**: Find optimal hyperplane
# MAGIC - **Pros**: Effective in high-dimensional spaces
# MAGIC - **Best for**: Small to medium datasets
# MAGIC
# MAGIC ### 4️⃣ Random Forest
# MAGIC - **Approach**: Ensemble of decision trees
# MAGIC - **Pros**: Handles non-linear relationships
# MAGIC - **Best for**: Complex patterns, feature importance
# MAGIC
# MAGIC ## Project Integration
# MAGIC We'll compare all four algorithms on a multi-class classification task!

# COMMAND ----------

# DBTITLE 1,Multi-Algorithm Text Classification
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import time

print("📄 Text Classification: Comparing ML Algorithms\n")

# Create a larger, multi-class dataset
class_data = {
    'text': [
        # Technology (0)
        'new smartphone release features latest processor',
        'artificial intelligence breakthrough in research',
        'software update fixes security vulnerabilities',
        'cloud computing platform launches new services',
        'quantum computer achieves milestone calculation',
        
        # Sports (1)
        'team wins championship after overtime thriller',
        'player breaks record with amazing performance',
        'coach discusses strategy after victory',
        'tournament begins with exciting matches',
        'athlete signs contract with professional team',
        
        # Business (2)
        'stock market reaches all time high today',
        'company reports strong quarterly earnings growth',
        'merger between corporations announced yesterday',
        'startup raises millions in funding round',
        'economic indicators show positive trends',
        
        # Entertainment (3)
        'movie breaks box office records worldwide',
        'celebrity announces new music album release',
        'television series wins multiple awards',
        'streaming platform launches original content',
        'actor stars in upcoming blockbuster film',
        
        # Health (4)
        'medical research discovers new treatment method',
        'vaccine shows promising results in trials',
        'health experts recommend lifestyle changes',
        'hospital introduces advanced surgical technique',
        'study reveals benefits of healthy diet',
    ],
    'category': [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4]
}

category_names = ['Technology', 'Sports', 'Business', 'Entertainment', 'Health']

df_multiclass = pd.DataFrame(class_data)
print(f"📖 Dataset: {len(df_multiclass)} documents, {len(category_names)} categories")
print(f"\nCategory distribution:")
for i, name in enumerate(category_names):
    count = sum(df_multiclass['category'] == i)
    print(f"   {i}. {name}: {count} documents")

# Split data
X_text = df_multiclass['text']
y_cat = df_multiclass['category']

X_train_text, X_test_text, y_train_cat, y_test_cat = train_test_split(
    X_text, y_cat, test_size=0.3, random_state=42, stratify=y_cat
)

print(f"\nTrain: {len(X_train_text)}, Test: {len(X_test_text)}")

# Define classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': LinearSVC(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each classifier
results = {}

print("\n" + "="*70)
print("🤖 Training and Evaluating Classifiers")
print("="*70)

for clf_name, classifier in classifiers.items():
    print(f"\n🔄 {clf_name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100, stop_words='english')),
        ('clf', classifier)
    ])
    
    # Train
    start_time = time.time()
    pipeline.fit(X_train_text, y_train_cat)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred_cat = pipeline.predict(X_test_text)
    pred_time = time.time() - start_time
    
    # Evaluate
    acc = accuracy_score(y_test_cat, y_pred_cat)
    
    results[clf_name] = {
        'accuracy': acc,
        'train_time': train_time,
        'pred_time': pred_time,
        'predictions': y_pred_cat
    }
    
    print(f"   ✅ Accuracy: {acc*100:.2f}%")
    print(f"   ⏱️  Training: {train_time*1000:.2f}ms, Prediction: {pred_time*1000:.2f}ms")

# Comparison table
print("\n" + "="*70)
print("📊 Results Comparison")
print("="*70 + "\n")

comparison_df = pd.DataFrame({
    'Algorithm': list(results.keys()),
    'Accuracy (%)': [results[k]['accuracy']*100 for k in results.keys()],
    'Train Time (ms)': [results[k]['train_time']*1000 for k in results.keys()],
    'Pred Time (ms)': [results[k]['pred_time']*1000 for k in results.keys()]
})

print(comparison_df.to_string(index=False))

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
ax1 = axes[0]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
ax1.barh(comparison_df['Algorithm'], comparison_df['Accuracy (%)'], color=colors)
ax1.set_xlabel('Accuracy (%)')
ax1.set_title('Accuracy Comparison')
ax1.set_xlim([0, 100])
for i, v in enumerate(comparison_df['Accuracy (%)']):
    ax1.text(v + 1, i, f'{v:.1f}%', va='center')

# Speed comparison
ax2 = axes[1]
ax2.barh(comparison_df['Algorithm'], comparison_df['Train Time (ms)'], color=colors, alpha=0.7, label='Training')
ax2.set_xlabel('Time (ms)')
ax2.set_title('Training Speed Comparison')
ax2.legend()

plt.tight_layout()
plt.show()

# Detailed report for best model
best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
print(f"\n🏆 Best Model: {best_model}\n")

# Get predictions from best model
best_pred = results[best_model]['predictions']
print("Classification Report:")
print(classification_report(y_test_cat, best_pred, target_names=category_names))

# Confusion matrix
cm = confusion_matrix(y_test_cat, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=category_names, yticklabels=category_names)
plt.title(f'Confusion Matrix - {best_model}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Test with new examples
test_texts = [
    "Scientists develop groundbreaking medical treatment",
    "Basketball team dominates in playoff game",
    "Tech startup receives investment funding"
]

print("\n🧪 Testing New Examples:\n")
pipeline_best = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100, stop_words='english')),
    ('clf', classifiers[best_model])
])
pipeline_best.fit(X_text, y_cat)

for text in test_texts:
    pred = pipeline_best.predict([text])[0]
    pred_proba = pipeline_best.predict_proba([text])[0] if hasattr(pipeline_best.named_steps['clf'], 'predict_proba') else None
    
    print(f"Text: '{text}'")
    print(f"Predicted: {category_names[pred]}")
    if pred_proba is not None:
        print(f"Confidence: {pred_proba[pred]*100:.1f}%")
    print()

print("✅ Text classification with multiple algorithms complete! 🎉")

# COMMAND ----------

# DBTITLE 1,🏷️ Named Entity Recognition (NER)
# MAGIC %md
# MAGIC # Cell 12: Named Entity Recognition (NER)
# MAGIC
# MAGIC ## Kya hai NER?
# MAGIC
# MAGIC Text se **important entities** extract karna:
# MAGIC
# MAGIC ```
# MAGIC "Apple Inc. CEO Tim Cook announced iPhone 15 in Cupertino, California."
# MAGIC
# MAGIC 🏛️ Organization: Apple Inc.
# MAGIC 👤 Person: Tim Cook  
# MAGIC 📦 Product: iPhone 15
# MAGIC 🌍 Location: Cupertino, California
# MAGIC ```
# MAGIC
# MAGIC ## Entity Types (Common):
# MAGIC
# MAGIC * **PERSON** - Names of people
# MAGIC * **ORG** - Organizations, companies
# MAGIC * **GPE** - Geo-political entities (cities, countries)
# MAGIC * **DATE** - Dates and time references
# MAGIC * **MONEY** - Monetary values
# MAGIC * **PRODUCT** - Products, objects
# MAGIC
# MAGIC ## Approaches:
# MAGIC
# MAGIC ### 1️⃣ Rule-Based
# MAGIC - Regex patterns
# MAGIC - Dictionaries / Gazetteers
# MAGIC - Fast but limited
# MAGIC
# MAGIC ### 2️⃣ ML-Based
# MAGIC - CRF (Conditional Random Fields)
# MAGIC - Features: POS tags, word shapes, context
# MAGIC - Better generalization
# MAGIC
# MAGIC ### 3️⃣ Deep Learning (later!)
# MAGIC - BiLSTM-CRF
# MAGIC - Transformers (BERT)
# MAGIC - State-of-the-art
# MAGIC
# MAGIC ## IOB Tagging Format:
# MAGIC
# MAGIC ```
# MAGIC Apple    B-ORG   (Begin Organization)
# MAGIC Inc.     I-ORG   (Inside Organization)
# MAGIC CEO      O       (Outside/Other)
# MAGIC Tim      B-PER   (Begin Person)
# MAGIC Cook     I-PER   (Inside Person)
# MAGIC ```
# MAGIC
# MAGIC ## Project Use
# MAGIC NER helps extract structured info from unstructured text!

# COMMAND ----------

# DBTITLE 1,NER Implementation with SpaCy
# MAGIC %pip install spacy -q
# MAGIC # Download small English model
# MAGIC import sys
# MAGIC !{sys.executable} -m spacy download en_core_web_sm -q
# MAGIC
# MAGIC import spacy
# MAGIC from collections import Counter
# MAGIC
# MAGIC print("🏷️ Named Entity Recognition with SpaCy\n")
# MAGIC
# MAGIC # Load SpaCy model
# MAGIC nlp = spacy.load('en_core_web_sm')
# MAGIC
# MAGIC print("✅ SpaCy model loaded!\n")
# MAGIC
# MAGIC # Sample texts
# MAGIC sample_texts = [
# MAGIC     "Apple Inc. CEO Tim Cook announced the new iPhone 15 in Cupertino, California on September 12, 2023.",
# MAGIC     "Elon Musk's company Tesla will open a new factory in Berlin, Germany next year.",
# MAGIC     "Microsoft acquired LinkedIn for $26.2 billion in 2016.",
# MAGIC     "The World Health Organization reported COVID-19 cases across 195 countries.",
# MAGIC     "Amazon founder Jeff Bezos stepped down as CEO in July 2021."
# MAGIC ]
# MAGIC
# MAGIC print("📝 Processing Texts for Named Entities:\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC all_entities = []
# MAGIC
# MAGIC for i, text in enumerate(sample_texts, 1):
# MAGIC     doc = nlp(text)
# MAGIC     
# MAGIC     print(f"\n{i}. Text: {text}")
# MAGIC     print(f"   Entities found:")
# MAGIC     
# MAGIC     if len(doc.ents) == 0:
# MAGIC         print("      No entities detected")
# MAGIC     else:
# MAGIC         for ent in doc.ents:
# MAGIC             print(f"      🔹 {ent.text:<25} ({ent.label_:<10}) - {spacy.explain(ent.label_)}")
# MAGIC             all_entities.append((ent.text, ent.label_))
# MAGIC
# MAGIC print("\n" + "="*80)
# MAGIC
# MAGIC # Entity statistics
# MAGIC entity_types = [ent[1] for ent in all_entities]
# MAGIC entity_counts = Counter(entity_types)
# MAGIC
# MAGIC print("\n📊 Entity Type Distribution:\n")
# MAGIC for ent_type, count in entity_counts.most_common():
# MAGIC     print(f"   {ent_type:<15} {count:>3} occurrences - {spacy.explain(ent_type)}")
# MAGIC
# MAGIC # Visualize with displaCy (text-based)
# MAGIC print("\n🎨 Detailed NER Visualization (Sample):\n")
# MAGIC
# MAGIC sample_doc = nlp(sample_texts[0])
# MAGIC print(f"Text: {sample_texts[0]}\n")
# MAGIC for token in sample_doc:
# MAGIC     ent_type = token.ent_type_ if token.ent_iob_ != 'O' else 'O'
# MAGIC     ent_iob = token.ent_iob_
# MAGIC     print(f"{token.text:<15} POS: {token.pos_:<8} ENT: {ent_iob}-{ent_type if ent_type != 'O' else ''}")
# MAGIC
# MAGIC # Custom NER for specific domain
# MAGIC print("\n" + "="*80)
# MAGIC print("🛠️ Custom Entity Extraction (Rule-Based)")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC import re
# MAGIC
# MAGIC def extract_custom_entities(text):
# MAGIC     """
# MAGIC     Custom rule-based entity extraction
# MAGIC     """
# MAGIC     entities = {
# MAGIC         'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
# MAGIC         'phones': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text),
# MAGIC         'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
# MAGIC         'money': re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
# MAGIC     }
# MAGIC     return entities
# MAGIC
# MAGIC custom_text = """
# MAGIC Contact us at support@company.com or call 555-123-4567.
# MAGIC Visit our website at https://www.example.com
# MAGIC Prices start at $29.99 and go up to $1,299.00
# MAGIC """
# MAGIC
# MAGIC print(f"Text: {custom_text}")
# MAGIC custom_ents = extract_custom_entities(custom_text)
# MAGIC
# MAGIC print("\nExtracted Entities:")
# MAGIC for ent_type, values in custom_ents.items():
# MAGIC     if values:
# MAGIC         print(f"   {ent_type.upper()}: {values}")
# MAGIC
# MAGIC # Entity-based text analysis
# MAGIC print("\n" + "="*80)
# MAGIC print("🔍 Entity-Based Text Analysis")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC # Analyze all sample texts
# MAGIC org_names = []
# MAGIC person_names = []
# MAGIC locations = []
# MAGIC
# MAGIC for text in sample_texts:
# MAGIC     doc = nlp(text)
# MAGIC     for ent in doc.ents:
# MAGIC         if ent.label_ == 'ORG':
# MAGIC             org_names.append(ent.text)
# MAGIC         elif ent.label_ == 'PERSON':
# MAGIC             person_names.append(ent.text)
# MAGIC         elif ent.label_ in ['GPE', 'LOC']:
# MAGIC             locations.append(ent.text)
# MAGIC
# MAGIC print(f"🏛️  Organizations mentioned: {org_names}")
# MAGIC print(f"👤 People mentioned: {person_names}")
# MAGIC print(f"🌍 Locations mentioned: {locations}")
# MAGIC
# MAGIC # Visualization
# MAGIC fig, ax = plt.subplots(figsize=(10, 6))
# MAGIC ent_types = list(entity_counts.keys())
# MAGIC ent_values = list(entity_counts.values())
# MAGIC
# MAGIC ax.bar(ent_types, ent_values, color='skyblue', alpha=0.7)
# MAGIC ax.set_xlabel('Entity Type')
# MAGIC ax.set_ylabel('Count')
# MAGIC ax.set_title('Named Entity Distribution')
# MAGIC ax.grid(axis='y', alpha=0.3)
# MAGIC plt.xticks(rotation=45)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC print("\n✅ Named Entity Recognition complete! 🎉")
# MAGIC print("💡 NER extracts structured info from unstructured text!")

# COMMAND ----------

# DBTITLE 1,💬 POS Tagging - Understanding Word Roles
# MAGIC %md
# MAGIC # Cell 13: Part-of-Speech (POS) Tagging
# MAGIC
# MAGIC ## Kya hai POS Tagging?
# MAGIC
# MAGIC Har word ki grammatical role identify karna:
# MAGIC
# MAGIC ```
# MAGIC "The quick brown fox jumps over the lazy dog."
# MAGIC
# MAGIC The     DET   (Determiner)
# MAGIC quick   ADJ   (Adjective)
# MAGIC brown   ADJ   (Adjective)
# MAGIC fox     NOUN  (Noun)
# MAGIC jumps   VERB  (Verb)
# MAGIC over    ADP   (Adposition/Preposition)
# MAGIC the     DET   (Determiner)
# MAGIC lazy    ADJ   (Adjective)
# MAGIC dog     NOUN  (Noun)
# MAGIC ```
# MAGIC
# MAGIC ## Common POS Tags:
# MAGIC
# MAGIC * **NOUN** - Person, place, thing (dog, city, love)
# MAGIC * **VERB** - Action or state (run, is, think)
# MAGIC * **ADJ** - Describes noun (quick, beautiful)
# MAGIC * **ADV** - Describes verb/adj (quickly, very)
# MAGIC * **DET** - Determines noun (the, a, this)
# MAGIC * **PRON** - Replaces noun (he, she, it)
# MAGIC * **ADP** - Preposition (in, on, at, with)
# MAGIC * **CONJ** - Connects words (and, but, or)
# MAGIC
# MAGIC ## Why Important?
# MAGIC
# MAGIC ✅ Helps disambiguate word meaning
# MAGIC ✅ Useful for lemmatization
# MAGIC ✅ Feature for NER and parsing
# MAGIC ✅ Foundation for syntax analysis
# MAGIC
# MAGIC ## Example: Word Sense Disambiguation
# MAGIC
# MAGIC ```
# MAGIC "I saw a saw" 
# MAGIC
# MAGIC I       PRON
# MAGIC saw     VERB    (past tense of see)
# MAGIC a       DET
# MAGIC saw     NOUN    (cutting tool)
# MAGIC ```
# MAGIC
# MAGIC ## Project Use
# MAGIC POS tags improve feature engineering for classification and NER!

# COMMAND ----------

# DBTITLE 1,POS Tagging Implementation
print("💬 Part-of-Speech (POS) Tagging\n")

# Sample sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is fascinating and challenging.",
    "I saw a saw in the workshop yesterday.",
    "She can can the can before dinner.",
    "Time flies like an arrow; fruit flies like a banana."
]

print("📝 POS Tagging Results:\n")
print("="*80)

for i, sentence in enumerate(sentences, 1):
    doc = nlp(sentence)
    
    print(f"\n{i}. {sentence}\n")
    print(f"   {'Token':<15} {'POS':<8} {'Tag':<8} {'Dependency':<12} {'Description':<30}")
    print("   " + "-"*75)
    
    for token in doc:
        print(f"   {token.text:<15} {token.pos_:<8} {token.tag_:<8} {token.dep_:<12} {spacy.explain(token.pos_)}")

print("\n" + "="*80)

# POS distribution analysis
print("\n📊 POS Tag Distribution:\n")

all_pos = []
for sentence in sentences:
    doc = nlp(sentence)
    all_pos.extend([token.pos_ for token in doc])

pos_counts = Counter(all_pos)
print(f"Total tokens analyzed: {len(all_pos)}\n")

for pos, count in pos_counts.most_common():
    percentage = (count / len(all_pos)) * 100
    print(f"   {pos:<10} {count:>3} ({percentage:>5.1f}%)  - {spacy.explain(pos)}")

# Visualize POS distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart
ax1 = axes[0]
pos_types = [p[0] for p in pos_counts.most_common()]
pos_values = [p[1] for p in pos_counts.most_common()]

ax1.bar(pos_types, pos_values, color='coral', alpha=0.7)
ax1.set_xlabel('POS Tag')
ax1.set_ylabel('Count')
ax1.set_title('POS Tag Distribution (Bar Chart)')
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Pie chart
ax2 = axes[1]
ax2.pie(pos_values, labels=pos_types, autopct='%1.1f%%', startangle=90)
ax2.set_title('POS Tag Distribution (Pie Chart)')

plt.tight_layout()
plt.show()

# Advanced: Extract specific patterns
print("\n" + "="*80)
print("🔍 Pattern Extraction using POS Tags")
print("="*80 + "\n")

def extract_noun_phrases(text):
    """
    Extract noun phrases (DET + ADJ* + NOUN)
    """
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

def extract_verb_phrases(text):
    """
    Extract main verbs and their objects
    """
    doc = nlp(text)
    verb_phrases = []
    for token in doc:
        if token.pos_ == 'VERB':
            phrase = [token.text]
            # Add direct objects
            for child in token.children:
                if child.dep_ in ['dobj', 'attr']:
                    phrase.append(child.text)
            verb_phrases.append(' '.join(phrase))
    return verb_phrases

analysis_text = "The brilliant scientist discovered a groundbreaking treatment for the rare disease."

print(f"Text: {analysis_text}\n")

noun_phrases = extract_noun_phrases(analysis_text)
verb_phrases = extract_verb_phrases(analysis_text)

print(f"Noun Phrases:")
for np in noun_phrases:
    print(f"   • {np}")

print(f"\nVerb Phrases:")
for vp in verb_phrases:
    print(f"   • {vp}")

# Application: Keyword extraction using POS
print("\n" + "="*80)
print("🏷️ Keyword Extraction using POS Tags")
print("="*80 + "\n")

def extract_keywords(text):
    """
    Extract keywords (nouns and adjectives)
    """
    doc = nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop:
            keywords.append(token.lemma_.lower())
    return list(set(keywords))

article = """
Artificial intelligence and machine learning are transforming modern healthcare.
Advanced algorithms analyze medical images with incredible accuracy.
Researchers develop innovative solutions for complex diagnostic challenges.
"""

keywords = extract_keywords(article)

print(f"Article: {article}\n")
print(f"Extracted Keywords ({len(keywords)}):")
print(f"   {', '.join(sorted(keywords))}")

print("\n✅ POS Tagging complete! 🎉")
print("💡 POS tags unlock grammatical structure for advanced NLP!")

# COMMAND ----------

# DBTITLE 1,🎭 Sentiment Analysis - Complete Pipeline
# MAGIC %md
# MAGIC # Cell 14: Sentiment Analysis - End-to-End Project
# MAGIC
# MAGIC ## Goal: Build Production-Ready Sentiment Classifier
# MAGIC
# MAGIC ### Problem Statement:
# MAGIC Customer reviews ko analyze karna aur sentiment detect karna:
# MAGIC - **Positive** 😊: Happy, satisfied customers
# MAGIC - **Negative** 😞: Unhappy, disappointed customers
# MAGIC - **Neutral** 😐: Mixed or factual reviews
# MAGIC
# MAGIC ## Complete Pipeline:
# MAGIC
# MAGIC ```
# MAGIC 1. Data Collection → Real product reviews
# MAGIC 2. Data Preprocessing → Clean + normalize
# MAGIC 3. Feature Engineering → TF-IDF + word embeddings
# MAGIC 4. Model Selection → Compare algorithms
# MAGIC 5. Hyperparameter Tuning → Grid search
# MAGIC 6. Evaluation → Multiple metrics
# MAGIC 7. Error Analysis → Understand failures
# MAGIC 8. Deployment Ready → Save model
# MAGIC ```
# MAGIC
# MAGIC ## Challenges:
# MAGIC
# MAGIC ⚠️ **Sarcasm**: "Great product, totally didn't break in 2 days" (Negative, but words are positive!)
# MAGIC ⚠️ **Negation**: "not good" vs "good" - very different!
# MAGIC ⚠️ **Context**: "This phone is sick!" (Positive in slang)
# MAGIC ⚠️ **Mixed sentiment**: "Good quality but expensive"
# MAGIC
# MAGIC ## Advanced Features:
# MAGIC
# MAGIC 1. **Emoji sentiment** - 😊 = positive, 😢 = negative
# MAGIC 2. **Exclamation marks** - !!! indicates strong emotion
# MAGIC 3. **Capital letters** - AMAZING = emphasis
# MAGIC 4. **Negation handling** - not, never, no before adjectives
# MAGIC
# MAGIC Let's build it! 🚀

# COMMAND ----------

# DBTITLE 1,Complete Sentiment Analysis System
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import joblib

print("🎭 Complete Sentiment Analysis Pipeline\n")
print("="*80)

# Step 1: Enhanced Dataset
print("\n📊 Step 1: Creating Enhanced Dataset\n")

enhanced_reviews = {
    'text': [
        # Positive
        'Absolutely love this product! Best purchase ever! 😊',
        'AMAZING quality! Exceeded all my expectations!!!',
        'Fantastic! Works perfectly. Highly recommend.',
        'Great value for money. Very satisfied.',
        'Excellent customer service and fast delivery.',
        'Outstanding product! Will definitely buy again.',
        'Perfect! Exactly what I needed.',
        'Wonderful experience. Five stars!',
        'Superb quality and design. Love it!',
        'Best in class. Totally worth it.',
        
        # Negative  
        'Terrible product. Complete waste of money! 😞',
        'WORST purchase ever! Broke in 2 days.',
        'Horrible quality. Very disappointed.',
        'Do NOT buy this. Total scam.',
        'Awful experience. Poor customer service.',
        'Useless product. Doesnt work at all.',
        'Disgusting quality. Want my money back.',
        'Failed to meet expectations. Very upset.',
        'Pathetic build quality. Avoid this.',
        'Extremely poor. Not recommended.',
        
        # Neutral / Mixed
        'Its okay. Nothing special.',
        'Average product. Does the job.',
        'Good quality but too expensive.',
        'Works fine but delivery was late.',
        'Decent but not as described.',
        'Meh. Could be better.',
        'Not bad, not great either.',
        'Fair price for what you get.',
        'Acceptable quality. Some issues.',
        'Mixed feelings about this product.',
    ],
    'sentiment': [
        1,1,1,1,1,1,1,1,1,1,  # Positive
        0,0,0,0,0,0,0,0,0,0,  # Negative
        2,2,2,2,2,2,2,2,2,2   # Neutral
    ]
}

df_sentiment = pd.DataFrame(enhanced_reviews)
sentiment_labels = ['Negative 😞', 'Positive 😊', 'Neutral 😐']

print(f"Dataset size: {len(df_sentiment)} reviews")
print(f"\nSentiment distribution:")
for i, label in enumerate(sentiment_labels):
    count = sum(df_sentiment['sentiment'] == i)
    print(f"   {label}: {count} ({count/len(df_sentiment)*100:.1f}%)")

# Step 2: Advanced Preprocessing
print("\n🧹 Step 2: Advanced Preprocessing\n")

def advanced_preprocess(text):
    """
    Enhanced preprocessing with sentiment-aware features
    """
    # Count special features before cleaning
    emoji_positive = len(re.findall(r'[😊😄😁❤️👍]', text))
    emoji_negative = len(re.findall(r'[😞😢😡👎]', text))
    exclamations = text.count('!')
    capitals = sum(1 for c in text if c.isupper())
    
    # Basic preprocessing
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    # Keep some punctuation for sentiment (! ?)
    text = re.sub(r'[^\w\s!?]', '', text)
    text = ' '.join(text.split())
    
    # Add feature markers
    if emoji_positive > 0:
        text += ' EMOJI_POSITIVE '
    if emoji_negative > 0:
        text += ' EMOJI_NEGATIVE '
    if exclamations > 2:
        text += ' HIGH_EXCITEMENT '
    if capitals > 5:
        text += ' EMPHASIS '
    
    return text

# Apply preprocessing
df_sentiment['processed'] = df_sentiment['text'].apply(advanced_preprocess)

print("Sample preprocessing:")
for i in range(3):
    print(f"\nOriginal:  {df_sentiment['text'][i]}")
    print(f"Processed: {df_sentiment['processed'][i]}")

# Step 3: Feature Engineering
print("\n" + "="*80)
print("🛠️ Step 3: Feature Engineering")
print("="*80 + "\n")

# Split data
X = df_sentiment['processed']
y = df_sentiment['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Create advanced TF-IDF features
tfidf_vectorizer_advanced = TfidfVectorizer(
    max_features=200,
    ngram_range=(1, 2),  # Unigrams + Bigrams
    min_df=1,
    sublinear_tf=True
)

X_train_tfidf = tfidf_vectorizer_advanced.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer_advanced.transform(X_test)

print(f"\nFeature matrix shape: {X_train_tfidf.shape}")
print(f"   Samples: {X_train_tfidf.shape[0]}")
print(f"   Features: {X_train_tfidf.shape[1]}")

# Step 4: Model Training with Hyperparameter Tuning
print("\n" + "="*80)
print("🤖 Step 4: Model Training & Hyperparameter Tuning")
print("="*80 + "\n")

# Define model and parameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

print("Performing Grid Search for Naive Bayes...")
base_model = MultinomialNB()
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train_tfidf, y_train)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_

# Step 5: Evaluation
print("\n" + "="*80)
print("📊 Step 5: Comprehensive Evaluation")
print("="*80 + "\n")

y_pred = best_model.predict(X_test_tfidf)
y_pred_proba = best_model.predict_proba(X_test_tfidf)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

print(f"Overall Performance:")
print(f"   Accuracy:  {acc*100:.2f}%")
print(f"   Precision: {prec*100:.2f}%")
print(f"   Recall:    {rec*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")

print(f"\nPer-Class Performance:")
print(classification_report(y_test, y_pred, target_names=sentiment_labels))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=sentiment_labels, yticklabels=sentiment_labels)
plt.title('Confusion Matrix - Sentiment Classifier')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.tight_layout()
plt.show()

# Step 6: Error Analysis
print("\n" + "="*80)
print("🔍 Step 6: Error Analysis")
print("="*80 + "\n")

X_test_list = X_test.tolist()
original_test = df_sentiment.iloc[X_test.index]['text'].tolist()

errors = []
for i, (true_label, pred_label, orig_text) in enumerate(zip(y_test, y_pred, original_test)):
    if true_label != pred_label:
        errors.append({
            'text': orig_text,
            'true': sentiment_labels[true_label],
            'predicted': sentiment_labels[pred_label],
            'confidence': y_pred_proba[i][pred_label]
        })

print(f"Total errors: {len(errors)} out of {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)\n")

if errors:
    print("Sample misclassifications:\n")
    for i, error in enumerate(errors[:3], 1):
        print(f"{i}. Text: '{error['text']}'")
        print(f"   True: {error['true']}, Predicted: {error['predicted']} (conf: {error['confidence']:.2f})")
        print()

# Step 7: Save Model
print("\n" + "="*80)
print("💾 Step 7: Saving Model for Production")
print("="*80 + "\n")

model_artifacts = {
    'model': best_model,
    'vectorizer': tfidf_vectorizer_advanced,
    'labels': sentiment_labels,
    'preprocess_func': advanced_preprocess
}

print("✅ Model artifacts ready for deployment!")
print("   - Trained classifier")
print("   - TF-IDF vectorizer")
print("   - Label mappings")
print("   - Preprocessing function")

# Test with new examples
print("\n" + "="*80)
print("🧪 Testing with New Examples")
print("="*80 + "\n")

new_reviews = [
    "This is absolutely AMAZING! Love it so much!!! 😊",
    "Terrible quality. Completely disappointed. 😞",
    "It's okay, nothing special really.",
    "Best product ever! Highly recommend!!!",
    "Worst purchase. Do not buy."
]

for review in new_reviews:
    processed = advanced_preprocess(review)
    features = tfidf_vectorizer_advanced.transform([processed])
    prediction = best_model.predict(features)[0]
    probabilities = best_model.predict_proba(features)[0]
    
    print(f"Review: '{review}'")
    print(f"Sentiment: {sentiment_labels[prediction]}")
    print(f"Confidence: {probabilities[prediction]*100:.1f}%")
    print(f"All probabilities: {', '.join([f'{sentiment_labels[i]}={p*100:.1f}%' for i, p in enumerate(probabilities)])}")
    print()

print("✅ Complete Sentiment Analysis Pipeline Ready! 🎉")

# COMMAND ----------

# DBTITLE 1,⚙️ Feature Engineering for NLP
# MAGIC %md
# MAGIC # Cell 15: Feature Engineering - Combining Techniques
# MAGIC
# MAGIC ## Beyond Basic Features
# MAGIC
# MAGIC Abhi tak humne dekhe:
# MAGIC - BoW, TF-IDF → Word frequency features
# MAGIC - Word embeddings → Semantic features
# MAGIC - POS tags → Grammatical features
# MAGIC
# MAGIC ## Advanced Feature Engineering:
# MAGIC
# MAGIC ### 1️⃣ Text Statistics
# MAGIC - Length: character count, word count
# MAGIC - Average word length
# MAGIC - Sentence count
# MAGIC - Punctuation density
# MAGIC
# MAGIC ### 2️⃣ Lexical Features
# MAGIC - Capital letter ratio
# MAGIC - Digit presence
# MAGIC - Special characters
# MAGIC - Vocabulary richness (unique words / total words)
# MAGIC
# MAGIC ### 3️⃣ Sentiment-Specific
# MAGIC - Positive/negative word count
# MAGIC - Emotion words (from lexicons)
# MAGIC - Negation presence
# MAGIC - Intensifiers (very, extremely)
# MAGIC
# MAGIC ### 4️⃣ Domain-Specific
# MAGIC - Technical terms count
# MAGIC - Jargon detection
# MAGIC - Brand/product mentions
# MAGIC
# MAGIC ## Feature Combination Strategy:
# MAGIC
# MAGIC ```
# MAGIC Final Features = [TF-IDF] + [Statistics] + [Embeddings] + [Domain]
# MAGIC ```
# MAGIC
# MAGIC ## Project Application
# MAGIC Combining features improves model performance by 10-20%!

# COMMAND ----------

# DBTITLE 1,Advanced Feature Engineering
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import string

print("⚙️ Advanced Feature Engineering\n")
print("="*80)

# Sample dataset
samples = [
    "AMAZING product!!! Really love it. Best purchase ever! 😊",
    "terrible waste money disappointed",
    "The product quality is decent. Price seems fair. Mixed feelings overall.",
    "WOW! Incredible! 10/10 would recommend!!!!",
    "poor quality broken arrived damaged"
]

labels = [1, 0, 2, 1, 0]  # 0=negative, 1=positive, 2=neutral

# Feature extraction function
def extract_advanced_features(text):
    """
    Extract multiple types of features from text
    """
    features = {}
    
    # 1. Text Statistics
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    
    # 2. Lexical Features
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['digit_count'] = sum(1 for c in text if c.isdigit())
    features['punctuation_count'] = sum(1 for c in text if c in string.punctuation)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # 3. Vocabulary Richness
    words = text.lower().split()
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
    
    # 4. Sentiment Indicators
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful']
    negative_words = ['bad', 'terrible', 'horrible', 'worst', 'hate', 'awful', 'poor', 'waste']
    
    features['positive_word_count'] = sum(1 for word in words if word in positive_words)
    features['negative_word_count'] = sum(1 for word in words if word in negative_words)
    features['has_negation'] = 1 if any(word in words for word in ['not', 'no', 'never', 'neither']) else 0
    
    # 5. Emphasis Indicators
    features['has_emoji'] = 1 if any(char in text for char in ['😊', '😞', '😍', '😡']) else 0
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))  # like "yessss"
    features['all_caps_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
    
    return features

# Extract features for all samples
print("📦 Extracting Features...\n")
feature_dicts = [extract_advanced_features(text) for text in samples]

# Convert to DataFrame
df_features = pd.DataFrame(feature_dicts)

print("Feature Matrix:")
print(df_features)

print(f"\n📐 Feature Statistics:")
print(df_features.describe())

# Visualize feature importance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Feature heatmap
ax1 = axes[0, 0]
sns.heatmap(df_features.T, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax1)
ax1.set_title('Feature Heatmap Across Samples')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Features')

# Plot 2: Feature distribution
ax2 = axes[0, 1]
df_features.mean().plot(kind='barh', ax=ax2, color='skyblue')
ax2.set_title('Average Feature Values')
ax2.set_xlabel('Mean Value')

# Plot 3: Correlation matrix
ax3 = axes[1, 0]
corr = df_features.corr()
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax3, annot=False)
ax3.set_title('Feature Correlation Matrix')

# Plot 4: Sample comparison
ax4 = axes[1, 1]
selected_features = ['word_count', 'positive_word_count', 'negative_word_count', 'exclamation_count']
df_features[selected_features].plot(kind='bar', ax=ax4)
ax4.set_title('Key Features Across Samples')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Value')
ax4.legend(loc='upper right', fontsize=8)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Combine with TF-IDF
print("\n" + "="*80)
print("🔧 Combining Multiple Feature Types")
print("="*80 + "\n")

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=20)
tfidf_features = vectorizer.fit_transform(samples)

# Statistical features
statistical_features = df_features.values

# Normalize statistical features
scaler = StandardScaler()
statistical_features_scaled = scaler.fit_transform(statistical_features)

# Combine features
from scipy.sparse import csr_matrix
combined_features = hstack([
    tfidf_features,
    csr_matrix(statistical_features_scaled)
])

print(f"TF-IDF shape: {tfidf_features.shape}")
print(f"Statistical features shape: {statistical_features_scaled.shape}")
print(f"Combined features shape: {combined_features.shape}")

print(f"\n💡 Feature engineering adds {statistical_features_scaled.shape[1]} extra features!")
print(f"   Total features: {combined_features.shape[1]}")

# Train models with and without extra features
print("\n" + "="*80)
print("⚖️ Comparing: TF-IDF Only vs Combined Features")
print("="*80 + "\n")

# Create larger dataset for meaningful comparison
large_dataset = df_sentiment.copy()
X_large = large_dataset['processed']
y_large = large_dataset['sentiment']

# Split
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
    X_large, y_large, test_size=0.25, random_state=42
)

# Extract features for train/test
X_train_original = large_dataset.iloc[X_train_large.index]['text'].tolist()
X_test_original = large_dataset.iloc[X_test_large.index]['text'].tolist()

# TF-IDF only
tfidf_vec = TfidfVectorizer(max_features=100)
X_train_tfidf_only = tfidf_vec.fit_transform(X_train_large)
X_test_tfidf_only = tfidf_vec.transform(X_test_large)

# Combined features
train_stat_features = StandardScaler().fit_transform(
    pd.DataFrame([extract_advanced_features(text) for text in X_train_original])
)
test_stat_features = StandardScaler().fit_transform(
    pd.DataFrame([extract_advanced_features(text) for text in X_test_original])
)

X_train_combined = hstack([X_train_tfidf_only, csr_matrix(train_stat_features)])
X_test_combined = hstack([X_test_tfidf_only, csr_matrix(test_stat_features)])

# Train models
model_tfidf = LogisticRegression(max_iter=1000, random_state=42)
model_combined = LogisticRegression(max_iter=1000, random_state=42)

model_tfidf.fit(X_train_tfidf_only, y_train_large)
model_combined.fit(X_train_combined, y_train_large)

# Evaluate
acc_tfidf = accuracy_score(y_test_large, model_tfidf.predict(X_test_tfidf_only))
acc_combined = accuracy_score(y_test_large, model_combined.predict(X_test_combined))

print(f"🎯 Results:")
print(f"   TF-IDF Only:       {acc_tfidf*100:.2f}%")
print(f"   Combined Features: {acc_combined*100:.2f}%")
print(f"   Improvement:       {(acc_combined - acc_tfidf)*100:+.2f}%")

print("\n✅ Feature engineering improves model performance! 🎉")

# COMMAND ----------

# DBTITLE 1,📊 Evaluation Metrics for NLP
# MAGIC %md
# MAGIC # Cell 16: Model Evaluation - Beyond Accuracy
# MAGIC
# MAGIC ## Why Accuracy Alone is Not Enough?
# MAGIC
# MAGIC ```
# MAGIC Dataset: 95 negative reviews, 5 positive reviews
# MAGIC Dumb model: Predict everything as "negative"
# MAGIC Accuracy: 95% ✅ but model is useless! ❌
# MAGIC ```
# MAGIC
# MAGIC ## Key Metrics:
# MAGIC
# MAGIC ### 1️⃣ Accuracy
# MAGIC ```
# MAGIC Accuracy = (TP + TN) / Total
# MAGIC ```
# MAGIC Good when classes are balanced
# MAGIC
# MAGIC ### 2️⃣ Precision
# MAGIC ```
# MAGIC Precision = TP / (TP + FP)
# MAGIC "Of all positive predictions, how many were correct?"
# MAGIC ```
# MAGIC Important when **false positives** are costly
# MAGIC
# MAGIC ### 3️⃣ Recall (Sensitivity)
# MAGIC ```
# MAGIC Recall = TP / (TP + FN)
# MAGIC "Of all actual positives, how many did we find?"
# MAGIC ```
# MAGIC Important when **false negatives** are costly
# MAGIC
# MAGIC ### 4️⃣ F1-Score
# MAGIC ```
# MAGIC F1 = 2 × (Precision × Recall) / (Precision + Recall)
# MAGIC ```
# MAGIC Harmonic mean of precision and recall
# MAGIC
# MAGIC ## Confusion Matrix:
# MAGIC
# MAGIC ```
# MAGIC                 Predicted
# MAGIC               Neg      Pos
# MAGIC Actual  Neg   TN       FP
# MAGIC         Pos   FN       TP
# MAGIC ```
# MAGIC
# MAGIC ## Multi-Class Metrics:
# MAGIC
# MAGIC - **Macro Average**: Average of per-class metrics (treats all classes equally)
# MAGIC - **Weighted Average**: Weighted by class support (accounts for imbalance)
# MAGIC - **Micro Average**: Global calculation across all classes
# MAGIC
# MAGIC ## NLP-Specific Metrics:
# MAGIC
# MAGIC - **BLEU**: Machine translation quality
# MAGIC - **ROUGE**: Summarization quality
# MAGIC - **Perplexity**: Language model quality
# MAGIC
# MAGIC We'll cover these in deep learning section!

# COMMAND ----------

# DBTITLE 1,Comprehensive Evaluation Framework
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

print("📊 Comprehensive Model Evaluation\n")
print("="*80)

# Use our sentiment model from earlier
print("\n📦 Using Sentiment Analysis Model...\n")

# Get predictions
y_pred = best_model.predict(X_test_tfidf)
y_pred_proba = best_model.predict_proba(X_test_tfidf)

# Calculate all metrics
print("📊 All Metrics:\n")

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Per-class metrics
for avg_type in ['macro', 'weighted', 'micro']:
    prec = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)
    
    print(f"\n{avg_type.capitalize()} Average:")
    print(f"   Precision: {prec*100:.2f}%")
    print(f"   Recall:    {rec*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")

# Detailed per-class report
print("\n" + "="*80)
print("📝 Detailed Per-Class Report")
print("="*80 + "\n")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=sentiment_labels, digits=3))

# Confusion Matrix with percentages
print("\n" + "="*80)
print("📊 Confusion Matrix Analysis")
print("="*80 + "\n")

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=sentiment_labels, yticklabels=sentiment_labels)
ax1.set_title('Confusion Matrix (Counts)')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# Normalized (percentages)
ax2 = axes[1]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
            xticklabels=sentiment_labels, yticklabels=sentiment_labels)
ax2.set_title('Confusion Matrix (Percentages)')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Per-class analysis
print("\nPer-Class Performance Analysis:\n")

for i, label in enumerate(sentiment_labels):
    # Get indices for this class
    class_indices = [idx for idx, val in enumerate(y_test) if val == i]
    
    if len(class_indices) == 0:
        continue
    
    # Calculate metrics for this class
    y_true_binary = [1 if val == i else 0 for val in y_test]
    y_pred_binary = [1 if val == i else 0 for val in y_pred]
    
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    support = len(class_indices)
    
    print(f"{label}:")
    print(f"   Support:   {support} samples")
    print(f"   Precision: {prec*100:.1f}% (of predicted {label}, how many were correct?)")
    print(f"   Recall:    {rec*100:.1f}% (of actual {label}, how many did we find?)")
    print(f"   F1-Score:  {f1*100:.1f}% (balance of precision and recall)")
    print()

# ROC Curve for multi-class
print("\n" + "="*80)
print("📊 ROC Curve (One-vs-Rest)")
print("="*80 + "\n")

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 7))
colors = ['red', 'green', 'blue']

for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{sentiment_labels[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Multi-Class Sentiment Classification')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print("\nROC AUC Scores:")
for i, label in enumerate(sentiment_labels):
    print(f"   {label}: {roc_auc[i]:.3f}")

# Error distribution analysis
print("\n" + "="*80)
print("🔍 Error Distribution Analysis")
print("="*80 + "\n")

# Calculate confidence for each prediction
max_probas = np.max(y_pred_proba, axis=1)

# Split into correct and incorrect
correct_mask = (y_test.values == y_pred)
correct_confidences = max_probas[correct_mask]
incorrect_confidences = max_probas[~correct_mask]

print(f"Correct predictions: {len(correct_confidences)} (avg confidence: {np.mean(correct_confidences):.3f})")
print(f"Incorrect predictions: {len(incorrect_confidences)} (avg confidence: {np.mean(incorrect_confidences):.3f})")

# Plot confidence distributions
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(correct_confidences, bins=10, alpha=0.7, label='Correct', color='green')
ax.hist(incorrect_confidences, bins=10, alpha=0.7, label='Incorrect', color='red')

ax.set_xlabel('Prediction Confidence')
ax.set_ylabel('Count')
ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 Key Insights:")
print("   - Correct predictions typically have higher confidence")
print("   - Low confidence predictions (<0.6) need review")
print("   - F1-score balances precision and recall")
print("   - Confusion matrix shows where model struggles")

print("\n✅ Comprehensive evaluation complete! 🎉")

# COMMAND ----------

# DBTITLE 1,⚖️ Handling Imbalanced Data in NLP
# MAGIC %md
# MAGIC # Cell 17: Handling Imbalanced Data
# MAGIC
# MAGIC ## Problem: Real-World Imbalance
# MAGIC
# MAGIC Real datasets rarely balanced hote hain:
# MAGIC
# MAGIC ```
# MAGIC Spam Detection:
# MAGIC - Ham (legitimate): 9,500 emails (95%)
# MAGIC - Spam: 500 emails (5%)
# MAGIC
# MAGIC Model that predicts everything as "Ham" = 95% accuracy! ❌
# MAGIC ```
# MAGIC
# MAGIC ## Techniques:
# MAGIC
# MAGIC ### 1️⃣ Resampling
# MAGIC
# MAGIC **Oversampling** (minority class):
# MAGIC - Random oversampling
# MAGIC - SMOTE (Synthetic Minority Over-sampling)
# MAGIC
# MAGIC **Undersampling** (majority class):
# MAGIC - Random undersampling
# MAGIC - Tomek links
# MAGIC
# MAGIC ### 2️⃣ Class Weights
# MAGIC ```python
# MAGIC class_weight='balanced'  # Automatically adjusts
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Ensemble Methods
# MAGIC - Balanced Random Forest
# MAGIC - EasyEnsemble
# MAGIC - RUSBoost
# MAGIC
# MAGIC ### 4️⃣ Threshold Adjustment
# MAGIC Change classification threshold from 0.5 to optimize F1
# MAGIC
# MAGIC ### 5️⃣ Evaluation Metrics
# MAGIC Use F1, Precision, Recall instead of accuracy
# MAGIC
# MAGIC ## Project Application
# MAGIC Imbalanced data is common in spam, fraud, rare disease detection!

# COMMAND ----------

# DBTITLE 1,Imbalanced Data Solutions
# MAGIC %pip install imbalanced-learn -q
# MAGIC
# MAGIC from imblearn.over_sampling import SMOTE, RandomOverSampler
# MAGIC from imblearn.under_sampling import RandomUnderSampler
# MAGIC from sklearn.utils.class_weight import compute_class_weight
# MAGIC
# MAGIC print("⚖️ Handling Imbalanced Text Data\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC # Create imbalanced dataset
# MAGIC imbalanced_data = {
# MAGIC     'text': [
# MAGIC         # Majority class (Legitimate - 85%)
# MAGIC         'meeting scheduled for tomorrow at 10am',
# MAGIC         'please review the attached document',
# MAGIC         'project deadline is next friday',
# MAGIC         'thank you for your prompt response',
# MAGIC         'quarterly report has been submitted',
# MAGIC         'team lunch on wednesday at noon',
# MAGIC         'client presentation went very well',
# MAGIC         'invoice has been processed successfully',
# MAGIC         'conference call rescheduled to 3pm',
# MAGIC         'budget approval pending from management',
# MAGIC         'new hire orientation next monday',
# MAGIC         'monthly newsletter sent to all staff',
# MAGIC         'office will be closed next thursday',
# MAGIC         'performance reviews due by month end',
# MAGIC         'training session recorded and shared',
# MAGIC         'database backup completed successfully',
# MAGIC         'system maintenance scheduled tonight',
# MAGIC         
# MAGIC         # Minority class (Spam - 15%)
# MAGIC         'URGENT winner claim your prize now',
# MAGIC         'click here for FREE money instantly',
# MAGIC         'you have won million dollars CONGRATULATIONS',
# MAGIC     ],
# MAGIC     'label': [0]*17 + [1]*3  # 0=legitimate, 1=spam
# MAGIC }
# MAGIC
# MAGIC df_imbalanced = pd.DataFrame(imbalanced_data)
# MAGIC
# MAGIC print("📊 Original Dataset Distribution:\n")
# MAGIC class_dist = df_imbalanced['label'].value_counts()
# MAGIC print(f"Legitimate (0): {class_dist[0]} ({class_dist[0]/len(df_imbalanced)*100:.1f}%)")
# MAGIC print(f"Spam (1):       {class_dist[1]} ({class_dist[1]/len(df_imbalanced)*100:.1f}%)")
# MAGIC print(f"\nImbalance Ratio: {class_dist[0]/class_dist[1]:.1f}:1")
# MAGIC
# MAGIC # Extract features
# MAGIC vectorizer_imb = TfidfVectorizer(max_features=50)
# MAGIC X_imb = vectorizer_imb.fit_transform(df_imbalanced['text'])
# MAGIC y_imb = df_imbalanced['label'].values
# MAGIC
# MAGIC # Split data
# MAGIC X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
# MAGIC     X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
# MAGIC )
# MAGIC
# MAGIC print(f"\nTrain set: {len(X_train_imb)} samples")
# MAGIC print(f"Test set:  {len(X_test_imb)} samples")
# MAGIC
# MAGIC # Baseline: Train without handling imbalance
# MAGIC print("\n" + "="*80)
# MAGIC print("🏁 Baseline: No Imbalance Handling")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC baseline_model = LogisticRegression(random_state=42, max_iter=1000)
# MAGIC baseline_model.fit(X_train_imb, y_train_imb)
# MAGIC y_pred_baseline = baseline_model.predict(X_test_imb)
# MAGIC
# MAGIC print("Results:")
# MAGIC print(f"   Accuracy:  {accuracy_score(y_test_imb, y_pred_baseline)*100:.2f}%")
# MAGIC print(f"   Precision: {precision_score(y_test_imb, y_pred_baseline, zero_division=0)*100:.2f}%")
# MAGIC print(f"   Recall:    {recall_score(y_test_imb, y_pred_baseline, zero_division=0)*100:.2f}%")
# MAGIC print(f"   F1-Score:  {f1_score(y_test_imb, y_pred_baseline, zero_division=0)*100:.2f}%")
# MAGIC
# MAGIC # Method 1: Class Weights
# MAGIC print("\n" + "="*80)
# MAGIC print("🎯 Method 1: Class Weights")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC # Calculate class weights
# MAGIC classes = np.unique(y_train_imb)
# MAGIC class_weights = compute_class_weight('balanced', classes=classes, y=y_train_imb)
# MAGIC class_weight_dict = dict(zip(classes, class_weights))
# MAGIC
# MAGIC print(f"Class weights: {class_weight_dict}")
# MAGIC print(f"   Minority class gets {class_weight_dict[1]/class_weight_dict[0]:.1f}x more weight\n")
# MAGIC
# MAGIC weighted_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
# MAGIC weighted_model.fit(X_train_imb, y_train_imb)
# MAGIC y_pred_weighted = weighted_model.predict(X_test_imb)
# MAGIC
# MAGIC print("Results:")
# MAGIC print(f"   Accuracy:  {accuracy_score(y_test_imb, y_pred_weighted)*100:.2f}%")
# MAGIC print(f"   Precision: {precision_score(y_test_imb, y_pred_weighted, zero_division=0)*100:.2f}%")
# MAGIC print(f"   Recall:    {recall_score(y_test_imb, y_pred_weighted, zero_division=0)*100:.2f}%")
# MAGIC print(f"   F1-Score:  {f1_score(y_test_imb, y_pred_weighted, zero_division=0)*100:.2f}%")
# MAGIC
# MAGIC # Method 2: SMOTE
# MAGIC print("\n" + "="*80)
# MAGIC print("🎯 Method 2: SMOTE (Synthetic Minority Oversampling)")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC smote = SMOTE(random_state=42)
# MAGIC X_train_smote, y_train_smote = smote.fit_resample(X_train_imb, y_train_imb)
# MAGIC
# MAGIC print(f"Before SMOTE: {Counter(y_train_imb)}")
# MAGIC print(f"After SMOTE:  {Counter(y_train_smote)}")
# MAGIC print(f"   Created {len(X_train_smote) - len(X_train_imb)} synthetic samples\n")
# MAGIC
# MAGIC smote_model = LogisticRegression(random_state=42, max_iter=1000)
# MAGIC smote_model.fit(X_train_smote, y_train_smote)
# MAGIC y_pred_smote = smote_model.predict(X_test_imb)
# MAGIC
# MAGIC print("Results:")
# MAGIC print(f"   Accuracy:  {accuracy_score(y_test_imb, y_pred_smote)*100:.2f}%")
# MAGIC print(f"   Precision: {precision_score(y_test_imb, y_pred_smote, zero_division=0)*100:.2f}%")
# MAGIC print(f"   Recall:    {recall_score(y_test_imb, y_pred_smote, zero_division=0)*100:.2f}%")
# MAGIC print(f"   F1-Score:  {f1_score(y_test_imb, y_pred_smote, zero_division=0)*100:.2f}%")
# MAGIC
# MAGIC # Method 3: Random Oversampling
# MAGIC print("\n" + "="*80)
# MAGIC print("🎯 Method 3: Random Oversampling")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC ros = RandomOverSampler(random_state=42)
# MAGIC X_train_ros, y_train_ros = ros.fit_resample(X_train_imb, y_train_imb)
# MAGIC
# MAGIC print(f"Before: {Counter(y_train_imb)}")
# MAGIC print(f"After:  {Counter(y_train_ros)}\n")
# MAGIC
# MAGIC ros_model = LogisticRegression(random_state=42, max_iter=1000)
# MAGIC ros_model.fit(X_train_ros, y_train_ros)
# MAGIC y_pred_ros = ros_model.predict(X_test_imb)
# MAGIC
# MAGIC print("Results:")
# MAGIC print(f"   Accuracy:  {accuracy_score(y_test_imb, y_pred_ros)*100:.2f}%")
# MAGIC print(f"   Precision: {precision_score(y_test_imb, y_pred_ros, zero_division=0)*100:.2f}%")
# MAGIC print(f"   Recall:    {recall_score(y_test_imb, y_pred_ros, zero_division=0)*100:.2f}%")
# MAGIC print(f"   F1-Score:  {f1_score(y_test_imb, y_pred_ros, zero_division=0)*100:.2f}%")
# MAGIC
# MAGIC # Comparison
# MAGIC print("\n" + "="*80)
# MAGIC print("📊 Comparison of All Methods")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC results_comparison = pd.DataFrame({
# MAGIC     'Method': ['Baseline', 'Class Weights', 'SMOTE', 'Random Oversampling'],
# MAGIC     'Accuracy': [
# MAGIC         accuracy_score(y_test_imb, y_pred_baseline),
# MAGIC         accuracy_score(y_test_imb, y_pred_weighted),
# MAGIC         accuracy_score(y_test_imb, y_pred_smote),
# MAGIC         accuracy_score(y_test_imb, y_pred_ros)
# MAGIC     ],
# MAGIC     'Precision': [
# MAGIC         precision_score(y_test_imb, y_pred_baseline, zero_division=0),
# MAGIC         precision_score(y_test_imb, y_pred_weighted, zero_division=0),
# MAGIC         precision_score(y_test_imb, y_pred_smote, zero_division=0),
# MAGIC         precision_score(y_test_imb, y_pred_ros, zero_division=0)
# MAGIC     ],
# MAGIC     'Recall': [
# MAGIC         recall_score(y_test_imb, y_pred_baseline, zero_division=0),
# MAGIC         recall_score(y_test_imb, y_pred_weighted, zero_division=0),
# MAGIC         recall_score(y_test_imb, y_pred_smote, zero_division=0),
# MAGIC         recall_score(y_test_imb, y_pred_ros, zero_division=0)
# MAGIC     ],
# MAGIC     'F1': [
# MAGIC         f1_score(y_test_imb, y_pred_baseline, zero_division=0),
# MAGIC         f1_score(y_test_imb, y_pred_weighted, zero_division=0),
# MAGIC         f1_score(y_test_imb, y_pred_smote, zero_division=0),
# MAGIC         f1_score(y_test_imb, y_pred_ros, zero_division=0)
# MAGIC     ]
# MAGIC })
# MAGIC
# MAGIC for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
# MAGIC     results_comparison[col] = (results_comparison[col] * 100).round(2)
# MAGIC
# MAGIC print(results_comparison.to_string(index=False))
# MAGIC
# MAGIC # Visualize
# MAGIC fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# MAGIC
# MAGIC metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
# MAGIC for idx, metric in enumerate(metrics):
# MAGIC     ax = axes[idx // 2, idx % 2]
# MAGIC     ax.bar(results_comparison['Method'], results_comparison[metric], color=['gray', 'blue', 'green', 'orange'])
# MAGIC     ax.set_ylabel(f'{metric} (%)')
# MAGIC     ax.set_title(f'{metric} Comparison')
# MAGIC     ax.set_ylim([0, 100])
# MAGIC     ax.tick_params(axis='x', rotation=45)
# MAGIC     for i, v in enumerate(results_comparison[metric]):
# MAGIC         ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
# MAGIC
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC print("\n💡 Key Insights:")
# MAGIC print("   - Imbalanced data hurts minority class recall")
# MAGIC print("   - Class weights: Easy to implement, no data modification")
# MAGIC print("   - SMOTE: Creates synthetic samples intelligently")
# MAGIC print("   - Choose method based on F1-score and business needs")
# MAGIC
# MAGIC print("\n✅ Imbalanced data handling complete! 🎉")

# COMMAND ----------

# DBTITLE 1,♻️ Cross-Validation for Text Data
# MAGIC %md
# MAGIC # Cell 18: Cross-Validation - Robust Evaluation
# MAGIC
# MAGIC ## Problem: Single Train-Test Split
# MAGIC
# MAGIC Ek hi split se results biased ho sakte hain:
# MAGIC
# MAGIC ```
# MAGIC Lucky split → High accuracy (but not generalizable)
# MAGIC Unlucky split → Low accuracy (but model is actually good)
# MAGIC ```
# MAGIC
# MAGIC ## Solution: Cross-Validation
# MAGIC
# MAGIC ### K-Fold Cross-Validation:
# MAGIC
# MAGIC ```
# MAGIC Data ko K parts mein divide karo:
# MAGIC
# MAGIC Fold 1: [Train | Train | Train | Test | Train]
# MAGIC Fold 2: [Train | Train | Test | Train | Train]  
# MAGIC Fold 3: [Train | Test | Train | Train | Train]
# MAGIC Fold 4: [Test | Train | Train | Train | Train]
# MAGIC Fold 5: [Train | Train | Train | Train | Test]
# MAGIC
# MAGIC Final score = Average of all folds
# MAGIC ```
# MAGIC
# MAGIC ## Types:
# MAGIC
# MAGIC ### 1️⃣ K-Fold
# MAGIC - Split into K equal parts
# MAGIC - Each part becomes test set once
# MAGIC - Good for balanced data
# MAGIC
# MAGIC ### 2️⃣ Stratified K-Fold
# MAGIC - Maintains class distribution
# MAGIC - **Best for classification**
# MAGIC - Ensures each fold is representative
# MAGIC
# MAGIC ### 3️⃣ Leave-One-Out (LOO)
# MAGIC - K = n (dataset size)
# MAGIC - Very thorough but expensive
# MAGIC - Good for small datasets
# MAGIC
# MAGIC ### 4️⃣ Time Series Split
# MAGIC - For temporal data
# MAGIC - Train on past, test on future
# MAGIC - Avoids data leakage
# MAGIC
# MAGIC ## Benefits:
# MAGIC
# MAGIC ✅ More reliable performance estimate
# MAGIC ✅ Uses all data for both training and testing
# MAGIC ✅ Helps detect overfitting
# MAGIC ✅ Reduces variance in results
# MAGIC
# MAGIC ## Project Use
# MAGIC Always use cross-validation before final model selection!

# COMMAND ----------

# DBTITLE 1,Cross-Validation Implementation
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, LeaveOneOut
)

print("♻️ Cross-Validation for Text Classification\n")
print("="*80)

# Use sentiment dataset
print("📊 Dataset: Sentiment Analysis")
print(f"   Total samples: {len(df_sentiment)}")
print(f"   Classes: {sentiment_labels}\n")

# Prepare features
X_cv = tfidf_vectorizer_advanced.fit_transform(df_sentiment['processed'])
y_cv = df_sentiment['sentiment']

print(f"Feature matrix: {X_cv.shape}\n")

# Method 1: Simple K-Fold
print("="*80)
print("🎯 Method 1: K-Fold Cross-Validation")
print("="*80 + "\n")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model_cv = MultinomialNB()

scores_kfold = cross_val_score(model_cv, X_cv, y_cv, cv=kfold, scoring='accuracy')

print(f"5-Fold Cross-Validation Results:\n")
for i, score in enumerate(scores_kfold, 1):
    print(f"   Fold {i}: {score*100:.2f}%")

print(f"\n   Mean:    {scores_kfold.mean()*100:.2f}%")
print(f"   Std Dev: {scores_kfold.std()*100:.2f}%")
print(f"   Range:   [{scores_kfold.min()*100:.2f}% - {scores_kfold.max()*100:.2f}%]")

# Method 2: Stratified K-Fold (RECOMMENDED for classification)
print("\n" + "="*80)
print("🎯 Method 2: Stratified K-Fold (Recommended)")
print("="*80 + "\n")

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Multiple metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

scores_stratified = cross_validate(
    model_cv, X_cv, y_cv,
    cv=skfold,
    scoring=scoring,
    return_train_score=True
)

print("Stratified 5-Fold Results:\n")

metric_names = {
    'test_accuracy': 'Accuracy',
    'test_precision_macro': 'Precision',
    'test_recall_macro': 'Recall',
    'test_f1_macro': 'F1-Score'
}

for key, name in metric_names.items():
    scores = scores_stratified[key]
    print(f"{name}:")
    print(f"   Mean: {scores.mean()*100:.2f}% (±{scores.std()*100:.2f}%)")
    print(f"   Folds: {[f'{s*100:.1f}%' for s in scores]}")
    print()

# Compare train vs test scores (overfitting check)
print("\n" + "="*80)
print("🔍 Overfitting Check: Train vs Test Scores")
print("="*80 + "\n")

train_acc = scores_stratified['train_accuracy'].mean()
test_acc = scores_stratified['test_accuracy'].mean()

print(f"Training Accuracy:   {train_acc*100:.2f}%")
print(f"Validation Accuracy: {test_acc*100:.2f}%")
print(f"Difference:          {(train_acc - test_acc)*100:.2f}%")

if (train_acc - test_acc) > 0.1:
    print("\n⚠️  Warning: Large gap suggests overfitting!")
    print("   Consider: regularization, simpler model, more data")
else:
    print("\n✅ Good generalization! No significant overfitting.")

# Visualize fold-by-fold performance
print("\n" + "="*80)
print("📊 Fold-by-Fold Performance Visualization")
print("="*80 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (key, name) in enumerate(metric_names.items()):
    ax = axes[idx // 2, idx % 2]
    
    train_key = key.replace('test_', 'train_')
    test_scores = scores_stratified[key] * 100
    train_scores = scores_stratified[train_key] * 100
    
    folds = range(1, 6)
    width = 0.35
    
    ax.bar([f - width/2 for f in folds], train_scores, width, label='Train', color='lightblue', alpha=0.8)
    ax.bar([f + width/2 for f in folds], test_scores, width, label='Test', color='coral', alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel(f'{name} (%)')
    ax.set_title(f'{name} Across Folds')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add mean lines
    ax.axhline(test_scores.mean(), color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Mean Test: {test_scores.mean():.1f}%')

plt.tight_layout()
plt.show()

# Method 3: Compare different K values
print("\n" + "="*80)
print("🔍 Effect of K (Number of Folds)")
print("="*80 + "\n")

k_values = [3, 5, 7, 10]
results_k = []

for k in k_values:
    skfold_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores_k = cross_val_score(model_cv, X_cv, y_cv, cv=skfold_k, scoring='f1_macro')
    results_k.append({
        'K': k,
        'Mean F1': scores_k.mean(),
        'Std F1': scores_k.std()
    })

df_k_comparison = pd.DataFrame(results_k)
df_k_comparison['Mean F1'] = (df_k_comparison['Mean F1'] * 100).round(2)
df_k_comparison['Std F1'] = (df_k_comparison['Std F1'] * 100).round(2)

print(df_k_comparison.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.errorbar(df_k_comparison['K'], df_k_comparison['Mean F1'], 
             yerr=df_k_comparison['Std F1'], marker='o', markersize=8,
             capsize=5, capthick=2, linewidth=2, color='purple')
plt.xlabel('Number of Folds (K)')
plt.ylabel('F1-Score (%)')
plt.title('Cross-Validation Performance vs K')
plt.grid(alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# Model Comparison using CV
print("\n" + "="*80)
print("🎯 Model Comparison with Cross-Validation")
print("="*80 + "\n")

models_to_compare = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': LinearSVC(random_state=42, max_iter=1000)
}

skfold_comp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison_results = []

for name, model in models_to_compare.items():
    scores = cross_val_score(model, X_cv, y_cv, cv=skfold_comp, scoring='f1_macro')
    comparison_results.append({
        'Model': name,
        'Mean F1': scores.mean(),
        'Std F1': scores.std(),
        'Min F1': scores.min(),
        'Max F1': scores.max()
    })

df_model_comparison = pd.DataFrame(comparison_results)
for col in ['Mean F1', 'Std F1', 'Min F1', 'Max F1']:
    df_model_comparison[col] = (df_model_comparison[col] * 100).round(2)

print(df_model_comparison.to_string(index=False))

# Visualize model comparison
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(df_model_comparison))
plt.bar(x_pos, df_model_comparison['Mean F1'], 
        yerr=df_model_comparison['Std F1'],
        capsize=5, color=['skyblue', 'lightgreen', 'coral'], alpha=0.8)
plt.xticks(x_pos, df_model_comparison['Model'])
plt.ylabel('F1-Score (%)')
plt.title('Model Comparison (5-Fold Cross-Validation)')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(df_model_comparison['Mean F1']):
    plt.text(i, v + df_model_comparison['Std F1'].iloc[i] + 2, 
             f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n💡 Best Practices:")
print("   - Always use StratifiedKFold for classification")
print("   - K=5 or K=10 are typical choices")
print("   - Report mean ± std for transparency")
print("   - Check train-test gap for overfitting")
print("   - Use CV for model selection, then train on full data")

print("\n✅ Cross-validation complete! Part 2 finished! 🎉")
print("🚀 Ready for Deep Learning (Part 3)! 🚀")

# COMMAND ----------

# DBTITLE 1,🤖 PART 3: Deep Learning for NLP
# MAGIC %md
# MAGIC # 🧠 PART 3: Deep Learning for NLP
# MAGIC
# MAGIC ## The Deep Learning Revolution
# MAGIC
# MAGIC Traditional ML → Manual feature engineering
# MAGIC Deep Learning → **Automatic** feature learning! 🎉
# MAGIC
# MAGIC ### What We'll Build:
# MAGIC
# MAGIC 1. **RNNs** - Sequential processing
# MAGIC 2. **LSTM/GRU** - Long-term memory
# MAGIC 3. **Bidirectional RNNs** - Context from both sides
# MAGIC 4. **Seq2Seq** - Sequence-to-sequence models
# MAGIC 5. **Attention** - The game changer! ✨
# MAGIC 6. **Transformers** - Modern architecture
# MAGIC
# MAGIC ### Why Deep Learning for NLP?
# MAGIC
# MAGIC ✅ **End-to-end learning** - No manual features
# MAGIC ✅ **Better representations** - Learned embeddings
# MAGIC ✅ **Captures context** - Sequential understanding
# MAGIC ✅ **State-of-the-art** - Best performance
# MAGIC
# MAGIC ### The Journey:
# MAGIC
# MAGIC ```
# MAGIC Feedforward NN → RNN → LSTM → Attention → Transformers
# MAGIC (No memory)   (Short)  (Better)  (Focus)    (Parallel)
# MAGIC ```
# MAGIC
# MAGIC ### Requirements:
# MAGIC
# MAGIC ```python
# MAGIC # We'll use PyTorch/Keras
# MAGIC %pip install torch tensorflow
# MAGIC ```
# MAGIC
# MAGIC Let's build neural networks! 🚀

# COMMAND ----------

# DBTITLE 1,🔄 RNN Fundamentals - Sequence Modeling
# MAGIC %md
# MAGIC # Cell 19: Recurrent Neural Networks (RNN)
# MAGIC
# MAGIC ## Problem: Feedforward NNs Don't Remember
# MAGIC
# MAGIC Normal neural networks:
# MAGIC - Process each input independently
# MAGIC - No memory of previous inputs
# MAGIC - Can't handle sequences!
# MAGIC
# MAGIC ```
# MAGIC "I love this" → [Process] → Positive ✅
# MAGIC "I don't love this" → [Process] → Still Positive? ❌
# MAGIC ```
# MAGIC
# MAGIC ## Solution: Recurrent Neural Networks
# MAGIC
# MAGIC ### Key Idea: **Hidden State** (Memory)
# MAGIC
# MAGIC ```
# MAGIC h(t) = f(h(t-1), x(t))
# MAGIC
# MAGIC Current state = f(Previous state, Current input)
# MAGIC ```
# MAGIC
# MAGIC ### RNN Architecture:
# MAGIC
# MAGIC ```
# MAGIC Input:  x1 → x2 → x3 → x4
# MAGIC          ↓    ↓    ↓    ↓
# MAGIC Hidden: h1 → h2 → h3 → h4
# MAGIC          ↓    ↓    ↓    ↓
# MAGIC Output: y1   y2   y3   y4
# MAGIC ```
# MAGIC
# MAGIC ### Mathematical Formula:
# MAGIC
# MAGIC ```
# MAGIC h(t) = tanh(W_hh * h(t-1) + W_xh * x(t) + b_h)
# MAGIC y(t) = W_hy * h(t) + b_y
# MAGIC ```
# MAGIC
# MAGIC ## RNN Types:
# MAGIC
# MAGIC ### 1️⃣ One-to-Many
# MAGIC ```
# MAGIC Image → [RNN] → Caption (sequence of words)
# MAGIC ```
# MAGIC
# MAGIC ### 2️⃣ Many-to-One
# MAGIC ```
# MAGIC Sentence (sequence) → [RNN] → Sentiment (single label)
# MAGIC ```
# MAGIC
# MAGIC ### 3️⃣ Many-to-Many (same length)
# MAGIC ```
# MAGIC Words → [RNN] → POS tags (same length)
# MAGIC ```
# MAGIC
# MAGIC ### 4️⃣ Many-to-Many (different length)
# MAGIC ```
# MAGIC English → [RNN] → Hindi (translation)
# MAGIC ```
# MAGIC
# MAGIC ## Problem: Vanishing Gradients
# MAGIC
# MAGIC ```
# MAGIC Long sequences → Gradients shrink → Can't learn long dependencies ❌
# MAGIC
# MAGIC "The cat that ate the mouse that lived in the house was" [black/white]?
# MAGIC
# MAGIC RNN struggles to connect "cat" with "was"!
# MAGIC ```
# MAGIC
# MAGIC ## Project Application
# MAGIC We'll use RNN for sentiment analysis to see sequence modeling in action!

# COMMAND ----------

# DBTITLE 1,Simple RNN Implementation
# MAGIC %pip install torch torchtext -q
# MAGIC
# MAGIC import torch
# MAGIC import torch.nn as nn
# MAGIC import torch.optim as optim
# MAGIC from torch.utils.data import Dataset, DataLoader
# MAGIC import torch.nn.functional as F
# MAGIC
# MAGIC print("🔄 Recurrent Neural Network (RNN) Implementation\n")
# MAGIC print("="*80)
# MAGIC
# MAGIC # Check device
# MAGIC device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MAGIC print(f"🖥️  Device: {device}\n")
# MAGIC
# MAGIC # Simple RNN for sentiment analysis
# MAGIC class SimpleRNN(nn.Module):
# MAGIC     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
# MAGIC         super(SimpleRNN, self).__init__()
# MAGIC         
# MAGIC         # Embedding layer
# MAGIC         self.embedding = nn.Embedding(vocab_size, embedding_dim)
# MAGIC         
# MAGIC         # RNN layer
# MAGIC         self.rnn = nn.RNN(
# MAGIC             embedding_dim,
# MAGIC             hidden_dim,
# MAGIC             num_layers=n_layers,
# MAGIC             batch_first=True
# MAGIC         )
# MAGIC         
# MAGIC         # Fully connected layer
# MAGIC         self.fc = nn.Linear(hidden_dim, output_dim)
# MAGIC         
# MAGIC         self.hidden_dim = hidden_dim
# MAGIC         self.n_layers = n_layers
# MAGIC     
# MAGIC     def forward(self, text):
# MAGIC         # text shape: [batch_size, seq_len]
# MAGIC         
# MAGIC         # Embedding
# MAGIC         embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
# MAGIC         
# MAGIC         # RNN
# MAGIC         output, hidden = self.rnn(embedded)  # output: [batch_size, seq_len, hidden_dim]
# MAGIC         
# MAGIC         # Take last hidden state
# MAGIC         last_hidden = hidden[-1]  # [batch_size, hidden_dim]
# MAGIC         
# MAGIC         # Fully connected
# MAGIC         logits = self.fc(last_hidden)  # [batch_size, output_dim]
# MAGIC         
# MAGIC         return logits
# MAGIC
# MAGIC # Create simple vocabulary
# MAGIC sentences = [
# MAGIC     "i love this movie",
# MAGIC     "this is great",
# MAGIC     "i hate this",
# MAGIC     "terrible movie",
# MAGIC     "amazing film",
# MAGIC     "awful experience",
# MAGIC     "best movie ever",
# MAGIC     "worst film"
# MAGIC ]
# MAGIC labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
# MAGIC
# MAGIC # Build vocabulary
# MAGIC word_to_idx = {'<PAD>': 0, '<UNK>': 1}
# MAGIC for sent in sentences:
# MAGIC     for word in sent.split():
# MAGIC         if word not in word_to_idx:
# MAGIC             word_to_idx[word] = len(word_to_idx)
# MAGIC
# MAGIC idx_to_word = {idx: word for word, idx in word_to_idx.items()}
# MAGIC
# MAGIC vocab_size = len(word_to_idx)
# MAGIC print(f"📚 Vocabulary:")
# MAGIC print(f"   Size: {vocab_size}")
# MAGIC print(f"   Words: {list(word_to_idx.keys())[:15]}...\n")
# MAGIC
# MAGIC # Convert sentences to indices
# MAGIC def sent_to_indices(sentence, word_to_idx, max_len=10):
# MAGIC     indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in sentence.split()]
# MAGIC     # Pad to max_len
# MAGIC     if len(indices) < max_len:
# MAGIC         indices += [word_to_idx['<PAD>']] * (max_len - len(indices))
# MAGIC     else:
# MAGIC         indices = indices[:max_len]
# MAGIC     return indices
# MAGIC
# MAGIC X = [sent_to_indices(sent, word_to_idx) for sent in sentences]
# MAGIC y = labels
# MAGIC
# MAGIC X_tensor = torch.LongTensor(X)
# MAGIC y_tensor = torch.LongTensor(y)
# MAGIC
# MAGIC print(f"📐 Data Shape:")
# MAGIC print(f"   X: {X_tensor.shape} (batch_size, seq_len)")
# MAGIC print(f"   y: {y_tensor.shape}\n")
# MAGIC
# MAGIC print(f"Sample encoding:")
# MAGIC print(f"   Sentence: '{sentences[0]}'")
# MAGIC print(f"   Indices:  {X[0]}")
# MAGIC print(f"   Label:    {y[0]} (Positive)\n")
# MAGIC
# MAGIC # Initialize model
# MAGIC embedding_dim = 16
# MAGIC hidden_dim = 32
# MAGIC output_dim = 2  # Binary classification
# MAGIC
# MAGIC model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
# MAGIC model = model.to(device)
# MAGIC
# MAGIC print("🤖 RNN Model Architecture:\n")
# MAGIC print(model)
# MAGIC print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}\n")
# MAGIC
# MAGIC # Training
# MAGIC criterion = nn.CrossEntropyLoss()
# MAGIC optimizer = optim.Adam(model.parameters(), lr=0.01)
# MAGIC
# MAGIC X_train = X_tensor.to(device)
# MAGIC y_train = y_tensor.to(device)
# MAGIC
# MAGIC print("🎯 Training RNN...\n")
# MAGIC
# MAGIC epochs = 100
# MAGIC losses = []
# MAGIC
# MAGIC for epoch in range(epochs):
# MAGIC     model.train()
# MAGIC     
# MAGIC     # Forward pass
# MAGIC     outputs = model(X_train)
# MAGIC     loss = criterion(outputs, y_train)
# MAGIC     
# MAGIC     # Backward pass
# MAGIC     optimizer.zero_grad()
# MAGIC     loss.backward()
# MAGIC     optimizer.step()
# MAGIC     
# MAGIC     losses.append(loss.item())
# MAGIC     
# MAGIC     if (epoch + 1) % 20 == 0:
# MAGIC         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
# MAGIC
# MAGIC print("\n✅ Training complete!\n")
# MAGIC
# MAGIC # Plot training loss
# MAGIC plt.figure(figsize=(10, 5))
# MAGIC plt.plot(losses, linewidth=2, color='blue')
# MAGIC plt.xlabel('Epoch')
# MAGIC plt.ylabel('Loss')
# MAGIC plt.title('RNN Training Loss')
# MAGIC plt.grid(alpha=0.3)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC # Evaluate
# MAGIC model.eval()
# MAGIC with torch.no_grad():
# MAGIC     outputs = model(X_train)
# MAGIC     _, predicted = torch.max(outputs, 1)
# MAGIC     accuracy = (predicted == y_train).sum().item() / len(y_train)
# MAGIC     
# MAGIC print(f"🃊 Training Accuracy: {accuracy*100:.2f}%\n")
# MAGIC
# MAGIC # Test predictions
# MAGIC print("🧪 Testing RNN Predictions:\n")
# MAGIC
# MAGIC test_sentences = [
# MAGIC     "i love this",
# MAGIC     "i hate this",
# MAGIC     "great movie",
# MAGIC     "terrible film"
# MAGIC ]
# MAGIC
# MAGIC for test_sent in test_sentences:
# MAGIC     test_indices = sent_to_indices(test_sent, word_to_idx)
# MAGIC     test_tensor = torch.LongTensor([test_indices]).to(device)
# MAGIC     
# MAGIC     with torch.no_grad():
# MAGIC         output = model(test_tensor)
# MAGIC         probabilities = F.softmax(output, dim=1)
# MAGIC         prediction = torch.argmax(output, dim=1).item()
# MAGIC     
# MAGIC     sentiment = "Positive" if prediction == 1 else "Negative"
# MAGIC     confidence = probabilities[0][prediction].item()
# MAGIC     
# MAGIC     print(f"Text: '{test_sent}'")
# MAGIC     print(f"   Prediction: {sentiment}")
# MAGIC     print(f"   Confidence: {confidence*100:.1f}%")
# MAGIC     print(f"   Probabilities: Negative={probabilities[0][0]*100:.1f}%, Positive={probabilities[0][1]*100:.1f}%")
# MAGIC     print()
# MAGIC
# MAGIC # Visualize hidden states
# MAGIC print("="*80)
# MAGIC print("🔍 Visualizing RNN Hidden States")
# MAGIC print("="*80 + "\n")
# MAGIC
# MAGIC test_sent = sentences[0]
# MAGIC test_tensor = torch.LongTensor([sent_to_indices(test_sent, word_to_idx)]).to(device)
# MAGIC
# MAGIC # Get intermediate hidden states
# MAGIC model.eval()
# MAGIC with torch.no_grad():
# MAGIC     embedded = model.embedding(test_tensor)
# MAGIC     output, hidden = model.rnn(embedded)
# MAGIC     hidden_states = output[0].cpu().numpy()  # [seq_len, hidden_dim]
# MAGIC
# MAGIC print(f"Sentence: '{test_sent}'")
# MAGIC print(f"Hidden states shape: {hidden_states.shape}\n")
# MAGIC
# MAGIC # Plot hidden state evolution
# MAGIC plt.figure(figsize=(12, 6))
# MAGIC plt.imshow(hidden_states.T, aspect='auto', cmap='viridis')
# MAGIC plt.colorbar(label='Activation')
# MAGIC plt.xlabel('Token Position')
# MAGIC plt.ylabel('Hidden Dimension')
# MAGIC plt.title(f'RNN Hidden State Evolution: "{test_sent}"')
# MAGIC words = test_sent.split() + ['<PAD>'] * (hidden_states.shape[0] - len(test_sent.split()))
# MAGIC plt.xticks(range(len(words)), words, rotation=45)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC print("💡 Key Insights:")
# MAGIC print("   - RNN processes sequences step-by-step")
# MAGIC print("   - Hidden state carries information forward")
# MAGIC print("   - Each word updates the hidden state")
# MAGIC print("   - Final state used for classification")
# MAGIC print("   - Problem: Vanishing gradients for long sequences!")
# MAGIC
# MAGIC print("\n✅ RNN fundamentals complete! Next: LSTM! 🚀")