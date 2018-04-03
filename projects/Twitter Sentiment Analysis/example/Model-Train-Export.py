
# coding: utf-8

# In[1]:


import pandas as pd

# Read reviews from CSV
reviews = pd.read_csv('epinions.csv')
reviews = reviews.as_matrix()[:, :]
print("%d reviews in dataset" % len(reviews))
# print(reviews[:1])
print(type(reviews))

print("checkmark 1")


# In[2]:

import numpy as np

path = 'Twitter_Dataset_cleaned_2-pos&neg.csv'
df = pd.read_csv(path, encoding='utf-8')

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

reviews = df.as_matrix()[:,:]


subsection = int(len(reviews) / 50)
reviews = reviews[:subsection]
print("%d reviews in dataset" % len(reviews))

print("checkmark 2")


# In[5]:


import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation


# nltk.download()

# Create features
def features(sentence):
    
    sentence = re.sub('<[^>]*>', '', sentence)
    #Remove hyperlinks
    sentence = re.sub(r"http\S+", '', sentence, flags=re.MULTILINE)
    #Remove quotes
    sentence = re.sub(r'&amp;quot;|&amp;amp', '', sentence)
    #Remove citations
    sentence = re.sub(r'(@[a-zA-Z0-9])\w*', '', sentence)
    #Remove hashtags
    sentence = re.sub(r'(#[a-zA-Z0-9])\w*', '', sentence)
    #Remove tickers
    sentence = re.sub(r'\$[a-zA-Z0-9]*', '', sentence)
    #Remove numbers
    sentence = re.sub(r'[0-9]*','',sentence)
    

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', sentence.lower())
    sentence = re.sub('[\W]+', ' ', sentence.lower()) +        ' '.join(emoticons).replace('-', '')
    
    stop_words = stopwords.words('english') + list(punctuation)
    words = word_tokenize(sentence)
    words = [w.lower() for w in words]
    filtered = [w for w in words if w not in stop_words and not w.isdigit()]
    words = {}
    for word in filtered:
        if word in words:
            words[word] += 1.0
        else:
            words[word] = 1.0
    return words

print("checkmark 3")


# In[6]:


# Vectorize the features function
features = np.vectorize(features)
# Extract the features for the whole dataset
X = features(reviews[:, 1])
# Set the targets
y = reviews[:, 0]

print("checkmark 4")


# In[7]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# Create grid search
clf = Pipeline([("dct", DictVectorizer()), ("svc", LinearSVC())])
params = {
    "svc__C": [1e15, 1e13, 1e11, 1e9, 1e7, 1e5, 1e3, 1e1, 1e-1, 1e-3, 1e-5]
}
gs = GridSearchCV(clf, params, cv=10, verbose=1, n_jobs=-1)
gs.fit(X, y)
model = gs.best_estimator_

# Print results
print(model.score(X, y))
print("Optimized parameters: ", model)
print("Best CV score: ", gs.best_score_)

print("checkmark 5")


# In[ ]:


import coremltools

# Convert to CoreML model
coreml_model = coremltools.converters.sklearn.convert(model)
coreml_model.author = 'Cameron Deardorff'
# coreml_model.license = 'MIT'
coreml_model.short_description = 'Sentiment polarity LinearSVC.'
coreml_model.input_description['input'] = 'Features extracted from the text.'
coreml_model.output_description['classLabel'] = 'The most likely polarity (positive or negative), for the given input.'
coreml_model.output_description['classProbability'] = 'The probabilities for each class label, for the given input.'
coreml_model.save('SentimentPolarity.mlmodel')

