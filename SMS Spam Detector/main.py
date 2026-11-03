
import joblib
import re

model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("=" * 50)
print("SMS SPAM DETECTION SYSTEM")
print("=" * 50)

while True:
    message = input("\nEnter your SMS message: ").strip()

    if message.lower() == "exit":
        print("Program terminated.")
        break

    if not message:
        print("Please enter a message.")
        continue

    cleaned_message = clean_text(message)

    message_tfidf = vectorizer.transform([cleaned_message])

    prediction = model.predict(message_tfidf)[0]

    probabilities = model.predict_proba(message_tfidf)[0]

    confidence = probabilities[prediction] * 100

    if prediction == 1:
        print("\nResult: SPAM MESSAGE")
    else:
        print("\nResult: HAM (NORMAL MESSAGE)")

    print(f"Confidence: {confidence:.2f}%")
