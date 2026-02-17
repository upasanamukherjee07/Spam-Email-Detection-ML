
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])


data['label'] = data['label'].map({'ham':0, 'spam':1})


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

data['message'] = data['message'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LinearSVC()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nModel Ready! Enter your own messages.\n")


def predict_message(msg):
    msg = clean_text(msg)
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    if prediction[0] == 1:
        return "Spam ❌"
    else:
        return "Not Spam (Ham) ✅"

while True:
    user_msg = input("Enter a message (type 'exit' to stop): ")

    if user_msg.lower() == "exit":
        print("Program Stopped.")
        break

    result = predict_message(user_msg)
    print("Result:", result, "\n")
