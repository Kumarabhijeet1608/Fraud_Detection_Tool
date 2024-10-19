import joblib
import pandas as pd
import easyocr
import cv2
import matplotlib.pyplot as plt


def load_sentiment_model():
    grid_search = joblib.load('tools\\Limitedsentiment.pkl')
    vectorizer = joblib.load('tools\\Limitedvectorizer.pkl')
    return grid_search, vectorizer

# Function to preprocess text
def PreProcessText(review):
    return review.lower()

# Function to predict sentiment
def predict_sentiment(review, grid_search, vectorizer):
    review_processed = PreProcessText(review)
    review_vectorized = vectorizer.transform([review_processed])
    prediction = grid_search.predict(review_vectorized)
    return prediction[0]

# Function for OCR using EasyOCR
def ocr_with_easyocr(image_path):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        image = cv2.imread(image_path)
        results = reader.readtext(image)

        extracted_text = ""
        for (bbox, text, prob) in results:
            extracted_text += text + " "
        
        # Optionally display the image with recognized text boxes
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
       

        return extracted_text.strip()
    
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return ""

# Function to analyze sentiment directly from the extracted text
def analyze_sentiment(image_path):
    grid_search, vectorizer = load_sentiment_model()
    extracted_text = ocr_with_easyocr(image_path)
    if extracted_text:
        predicted_sentiment = predict_sentiment(extracted_text, grid_search, vectorizer)
        print(f"Predicted sentiment: {predicted_sentiment}")

        # Check sentiment and print appropriate message
        if predicted_sentiment == "positive":
            print("Potential Fraud")
        else:
            print("Not potential Fraud")
    else:
        print("No text extracted from the image.")