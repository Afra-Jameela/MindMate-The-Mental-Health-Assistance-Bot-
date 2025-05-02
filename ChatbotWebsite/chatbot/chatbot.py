import json
import random
import numpy as np
import nltk
import re
import pickle
import tensorflow as tf
import google.generativeai as genai
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load intents
with open("ChatbotWebsite/static/data/intents.json") as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()

# Preprocessing the dataset
words = []
classes = []
documents = []
ignore_words = ["?", "!", ",", "."]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Creating training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in word_patterns else 0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Building the Deep Learning Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(train_x, train_y, epochs=2, batch_size=5, verbose=1)
model.save("chatbot_model.h5")

# Load model
model = tf.keras.models.load_model("chatbot_model.h5")

# Configure Gemini API
genai.configure(api_key="AIzaSyCuOd1rq3UMIy7JWQEuwt47KXi3fjqTDCg")  
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# Function to preprocess input text
def preprocess_input(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

# Function to create a bag of words
def bow(sentence, words):
    sentence_words = preprocess_input(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Function to predict the intent
def predict_intent(text):
    bow_vector = bow(text, words)
    res = model.predict(np.array([bow_vector]))[0]
    confidence_threshold = 0.7  # Adjust based on accuracy needs
    if max(res) > confidence_threshold:
        return classes[np.argmax(res)]
    return None  # If confidence is low, fallback to Gemini API

# Function to format AI responses
def format_response(text):
    """
    Formats AI-generated responses into structured text with headings and bullet points.
    
    Parameters:
        text (str): Raw AI response.
    
    Returns:
        str: Formatted response.
    """
    # Convert headings (**Title** → Title with spacing)
    text = re.sub(r'\*\*(.*?)\*\*', r'\n\1\n', text)

    # Convert subheadings (* **Subheading:** → Subheading with spacing)
    text = re.sub(r'\* \*\*(.*?)\*\*:', r'\n\1\n', text)

    # Convert list items (* Item → - Item)
    text = re.sub(r'\* (?!\*)(.*?)', r'- \1', text)

    # Remove any remaining bold markers
    text = text.replace('**', '')

    # Ensure proper line spacing
    text = re.sub(r'\n+', '\n\n', text).strip()

    return text + "\n"  # Ensure a final line break for better output

# Function to get chatbot response
def get_response(user_input):
    """
    Handles user queries by generating responses either from the trained model or Gemini AI.

    Parameters:
        user_input (str): User input query.

    Returns:
        str: Formatted chatbot response.
    """
    intent = predict_intent(user_input)
    if intent:
        for i in intents["intents"]:
            if i["tag"] == intent:
                return format_response(random.choice(i["responses"]))
    
    # Fallback to Gemini AI if no intent is detected
    response = model_gemini.generate_content(user_input)
    if response and response.text:
        return format_response(response.text)
    
    return "I'm sorry, I couldn't find an answer.\n"

# Chatbot loop
if __name__ == "__main__":
    print("\n🤖 Chatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye! 👋\n")
            break
        print("\nChatbot:\n", get_response(user_input))
