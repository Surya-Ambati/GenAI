'''
# Tokenization in Natural Language Processing (NLP) is the process of breaking down text into smaller units called tokens.
# These tokens can be words, phrases, or even characters, depending on the level of granularity required for the analysis.

or

Tokenization is the process of converting a sequence of characters into a sequence of tokens.
Tokens are the building blocks of text data and can be words, subwords, or characters. 
Tokenization is a crucial step in NLP as it helps in breaking down text into manageable pieces for further processing.

Tokenization can be performed using various techniques, including whitespace tokenization, punctuation-based tokenization, and more advanced methods like subword tokenization (e.g., Byte Pair Encoding or WordPiece).
Tokenization is often the first step in NLP pipelines, as it prepares the text data for tasks such as text classification, sentiment analysis, and machine translation.

# Tokenization is essential for converting unstructured text data into structured data that can be processed by machine learning algorithms.
# It helps in identifying the boundaries of words and phrases, allowing for better understanding and analysis of the text.
# Tokenization also plays a crucial role in feature extraction, as it enables the creation of numerical representations of text data, such as Bag of Words or TF-IDF vectors.
# Tokenization is a fundamental step in NLP that sets the stage for more complex tasks, such as named entity recognition, part-of-speech tagging, and dependency parsing.
# It is a critical component of any NLP pipeline, as it allows for the transformation of raw text into a format that can be easily analyzed and understood by machines.

example_text = "Hello, world! This is a sample text for tokenization."
tokens = example_text.split()  # Simple whitespace tokenization
print(tokens)  # Output: ['Hello,', 'world!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization.']
# Output: ['Hello,', 'world!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization.']
# Tokenization is a crucial step in NLP as it helps in breaking down text into manageable pieces for further processing.

example for tokenization:
# 1. Word Tokenization: Splitting text into individual words.
example_text = "Hello, world! This is a sample text for tokenization."
tokens = example_text.split()  # Simple whitespace tokenization


# 2. Sentence Tokenization: Splitting text into individual sentences.
example for sentence tokenization:

import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models

What is Punkt?
# Punkt is a pre-trained unsupervised machine learning model for tokenizing text into sentences and words. It is part of the NLTK library and is widely used for sentence segmentation in natural language processing tasks.

from nltk.tokenize import sent_tokenize

example_text = "Hello, world! This is a sample text for tokenization."
tokens = example_text.split('. ')  # Simple sentence tokenization
print(tokens)  # Output: ['Hello, world!', 'This is a sample text for tokenization.']
# Output: ['Hello, world!', 'This is a sample text for tokenization.']
# Sentence tokenization is useful for tasks like summarization and sentiment analysis, where understanding the context of sentences is important.



# 3. Subword Tokenization: Splitting words into smaller units (e.g., Byte Pair Encoding).

example for subword tokenization:

import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models
from nltk.tokenize import word_tokenize

example_text = "Hello, world! This is a sample text for tokenization."
tokens = word_tokenize(example_text)  # NLTK word tokenization
print(tokens)  # Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# Subword tokenization is useful for handling out-of-vocabulary words and improving the performance of language models.



# 4. Character Tokenization: Splitting text into individual characters.

example for character tokenization:

import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models
from nltk.tokenize import word_tokenize

example_text = "Hello, world! This is a sample text for tokenization."
tokens = list(example_text)  # Character tokenization
print(tokens)  # Output: ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', ' ', 'T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 's', 'a', 'm', 'p', 'l', 'e', ' ', 't', 'e', 'x', 't', ' ', 'f', 'o', 'r', ' ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.',]
# Output: ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!', ' ', 'T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 's', 'a', 'm', 'p', 'l', 'e', ' ', 't', 'e', 'x', 't', ' ', 'f', 'o', 'r', ' ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']  
# Character tokenization is useful for tasks like text generation and language modeling, where understanding the structure of words is important.


# 5. Regular Expression Tokenization: Using regex patterns to define token boundaries.

example for regex tokenization:

import re
example_text = "Hello, world! This is a sample text for tokenization."
tokens = re.findall(r'\w+', example_text)  # Regex tokenization
print(tokens)  # Output: ['Hello', 'world', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization']
# Output: ['Hello', 'world', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization']
# Regular expression tokenization is useful for custom tokenization rules, such as handling special characters or specific patterns in text.


# 6. Whitespace Tokenization: Splitting text based on whitespace characters.

example for whitespace tokenization:

import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models
from nltk.tokenize import word_tokenize

example_text = "Hello, world! This is a sample text for tokenization."
tokens = example_text.split()  # Whitespace tokenization
print(tokens)  # Output: ['Hello,', 'world!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization.']
# Output: ['Hello,', 'world!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization.']
# Whitespace tokenization is a simple and efficient method for breaking text into words, but it may not handle punctuation or special characters effectively.



# 7. Punctuation-based Tokenization: Splitting text based on punctuation marks.

example for punctuation-based tokenization:

import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models

example_text = "Hello, world! This is a sample text for tokenization."
tokens = re.findall(r'\w+', example_text)  # Punctuation-based tokenization
print(tokens)  # Output: ['Hello', 'world', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization']
# Output: ['Hello', 'world', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization']
# Punctuation-based tokenization is useful for tasks where punctuation marks are significant, such as sentiment analysis or text classification.

# 8. NLTK Tokenization: Using the Natural Language Toolkit library for tokenization.

example for NLTK tokenization:

import nltk
nltk.download('punkt')  # Download the Punkt tokenizer models
from nltk.tokenize import word_tokenize, sent_tokenize

example_text = "Hello, world! This is a sample text for tokenization."
tokens = word_tokenize(example_text)  # NLTK word tokenization
print(tokens)  # Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']

# Sentence tokenization using NLTK
example_text = "Hello, world! This is a sample text for tokenization."
tokens = sent_tokenize(example_text)  # NLTK sentence tokenization
print(tokens)  # Output: ['Hello, world!', 'This is a sample text for tokenization.']
# Output: ['Hello, world!', 'This is a sample text for tokenization.']
# NLTK tokenization is a powerful and flexible method for breaking text into words and sentences, making it suitable for various NLP tasks.
# NLTK provides pre-trained models for tokenization, making it easy to handle different languages and text formats.


# 9. SpaCy Tokenization: Using the SpaCy library for tokenization.
example for SpaCy tokenization:
import spacy
nlp = spacy.load("en_core_web_sm")  # Load the English model
example_text = "Hello, world! This is a sample text for tokenization."
doc = nlp(example_text)  # Process the text with SpaCy
tokens = [token.text for token in doc]  # SpaCy tokenization
print(tokens)  # Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# SpaCy tokenization is efficient and provides additional features like part-of-speech tagging and named entity recognition, making it suitable for advanced NLP tasks.


# 10. Custom Tokenization: Defining custom rules for tokenization based on specific requirements.

example for custom tokenization:
import re
example_text = "Hello, world! This is a sample text for tokenization."
custom_tokenizer = re.compile(r'\w+|\S')  # Custom regex for tokenization
tokens = custom_tokenizer.findall(example_text)  # Custom tokenization
print(tokens)  # Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'sample', 'text', 'for', 'tokenization', '.']
# Custom tokenization allows for flexibility in defining token boundaries based on specific text characteristics or requirements.
# Custom tokenization is useful for handling domain-specific text or when standard tokenization methods do not meet the requirements of a particular task.


# 11. Tokenization in Machine Learning: Preparing text data for machine learning models.

example for tokenization in machine learning:
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

example_text = [
    "I love programming!",
    "Python is great for data science.",
    "Tokenization is essential for NLP.",
    "Machine learning is fascinating.",
    "Deep learning is a subset of machine learning."
]   
labels = [1, 1, 1, 0, 0]  # Labels for sentiment analysis (1: positive, 0: negative)
df = pd.DataFrame({'text': example_text, 'label': labels})  # Create a DataFrame
split_text = df['text'].apply(lambda x: re.findall(r'\w+', x))  # Tokenization
split_text = split_text.apply(lambda x: ' '.join(x))  # Join tokens back into a string
split_text = split_text.tolist()  # Convert to list
new_labels = df['label'].tolist()  # Convert labels to list
X_train, X_test, y_train, y_test = train_test_split(split_text, new_labels, test_size=0.2, random_state=42)  # Train-test split
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())  # Create a pipeline with CountVectorizer and Naive Bayes classifier
pipeline.fit(X_train, y_train)  # Train the model
predictions = pipeline.predict(X_test)  # Make predictions
accuracy = accuracy_score(y_test, predictions)  # Calculate accuracy
print(f"Accuracy: {accuracy:.2f}")  # Output: Accuracy: 1.00
# Output: Accuracy: 1.00

# Tokenization in machine learning is crucial for preparing text data for analysis and modeling.
# It helps in converting unstructured text data into structured data that can be processed by machine learning algorithms.
# Tokenization is often the first step in machine learning pipelines, as it allows for the extraction of features from text data.
# Tokenization is essential for tasks like text classification, sentiment analysis, and topic modeling, where understanding the structure of text data is important.



# 12. Tokenization in Deep Learning: Preparing text data for deep learning models.

example for tokenization in deep learning:
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score

example_text = [
    "I love programming!",
    "Python is great for data science.",
    "Tokenization is essential for NLP.",
    "Machine learning is fascinating.",
    "Deep learning is a subset of machine learning."
]
labels = [1, 1, 1, 0, 0]  # Labels for sentiment analysis (1: positive, 0: negative)
df = pd.DataFrame({'text': example_text, 'label': labels})  # Create a DataFrame
preprocessed_text = df['text'].apply(lambda x: re.findall(r'\w+', x))  # Tokenization
preprocessed_text = preprocessed_text.apply(lambda x: ' '.join(x))  # Join tokens back into a string
preprocessed_text = preprocessed_text.tolist()  # Convert to list
preprocessed_labels = df['label'].tolist()  # Convert labels to list
preprocessed_text = [text.lower() for text in preprocessed_text]  # Convert text to lowercase
split_text = train_test_split(preprocessed_text, preprocessed_labels, test_size=0.2, random_state=42)  # Train-test split
split_text = list(split_text)  # Convert to list
split_text = [text for text in split_text if text]  # Filter out empty strings


# 13. Tokenization in NLP Pipelines: Integrating tokenization into NLP workflows.
# 14. Tokenization in Text Processing: Preprocessing text data for analysis.    
# 15. Tokenization in Data Science: Preparing text data for data analysis and visualization.
# 16. Tokenization in Text Mining: Extracting meaningful information from text data.
# 17. Tokenization in Information Retrieval: Indexing and searching text data.
# 18. Tokenization in Sentiment Analysis: Analyzing sentiment in text data.
# 19. Tokenization in Text Classification: Classifying text data into categories.
# 20. Tokenization in Named Entity Recognition: Identifying entities in text data.
# 21. Tokenization in Topic Modeling: Identifying topics in text data.
# 22. Tokenization in Language Modeling: Building language models from text data.
# 23. Tokenization in Text Generation: Generating text data from models.
# 24. Tokenization in Machine Translation: Translating text data between languages.
# 25. Tokenization in Speech Recognition: Converting speech data into text data.
# 26. Tokenization in Chatbots: Processing user input in chatbot applications.
# 27. Tokenization in Conversational AI: Understanding user intent in conversational interfaces.
# 28. Tokenization in Text Summarization: Summarizing text data.
# 29. Tokenization in Text-to-Speech: Converting text data into speech data.
# 30. Tokenization in Speech-to-Text: Converting speech data into text data.
# 31. Tokenization in Text Analysis: Analyzing text data for insights.
# 32. Tokenization in Text Mining: Extracting information from text data.
# 33. Tokenization in Data Preprocessing: Preparing text data for analysis.
# 34. Tokenization in Data Cleaning: Cleaning text data for analysis.
# 35. Tokenization in Data Transformation: Transforming text data for analysis.

'''