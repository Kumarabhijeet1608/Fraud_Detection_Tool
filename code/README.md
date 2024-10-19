
# Problem Statement

### Set- A(II)
### AI/ML System for Detecting and Categorizing Fraudulent Online Platforms – Websites and Mobile Applications

#### **Objective:**
To develop a robust AI/ML-powered system designed to detect and categorize fraudulent online content, including fake websites and mobile applications. The system aims to enhance online security by providing reliable verification and fraud detection capabilities.

#### **Key Features:**
1. **Legitimacy Verification:**
   - The system will verify the authenticity of websites and mobile applications by analyzing domain information, SSL certificates, and other critical authentication indicators.
   
2. **Content Analysis with NLP and Image Recognition:**
   - It will leverage Natural Language Processing (NLP) and image recognition techniques to assess the authenticity of online advertisements and app store listings, ensuring users can trust the content they engage with.

3. **Cross-checking Contact Information:**
   - Customer care numbers and other contact details will be cross-referenced with a verified fraud database (provided by the Goa Police) to identify potential scams or suspicious activities.

4. **Adaptive Machine Learning:**
   - The system will continuously improve its fraud detection capabilities using machine learning models that adapt based on patterns previously identified as fraudulent, becoming more accurate and effective over time.

This solution offers a comprehensive approach to combating online fraud, ensuring safer browsing and app usage for users worldwide.

# Tools


## **1. URL Fraud Detection and Content Sentiment Analysis Tool**

### **Overview**

This repository provides a Python script for detecting potential fraudulent websites and analyzing the sentiment of website content. It combines URL feature extraction and sentiment analysis to assess the overall safety of a given URL.

The main script is located in `main.py`, and it includes functionality for:
- Verifying URLs based on multiple factors (e.g., domain length, presence of IP, HTTPS verification).
- Scraping website content and analyzing it to predict sentiment (positive/negative).
- Generating a final fraud detection report based on both URL features and content analysis.

## **How It Works**

### **1. URL Feature Extraction**
The system extracts a variety of features from the URL to determine potential risks. These features include:
- IP presence in the URL
- Length of the URL
- Redirect behavior
- Subdomains and prefix-suffix analysis
- HTTPS usage
- Domain registration length
- Favicon location
- Website forwarding
- Google Index status, and more.

### **2. Sentiment Analysis**
The system uses a pre-trained machine learning model to predict the sentiment of the content (e.g., customer reviews, advertisements) found on the website. It evaluates whether the tone of the content is positive or negative, which may indicate whether a website is trustworthy.

### **3. Named Entity Recognition (NER)**
The script also analyzes the website content to extract named entities such as phone numbers and emails, helping to identify potential fraud based on suspicious or repeated contact details.

### **4. Combining Results**
The final fraud assessment is based on:
- **URL Feature Score**: Based on the extracted URL features.
- **Sentiment Analysis Score**: Based on the predicted sentiment of the website content.
- **NER Score**: Based on the presence of named entities like suspicious phone numbers or emails.

These scores are weighted to produce a final result, indicating whether the URL and its content are likely safe or risky.

### **Usage**

To use the script, run the following command from the terminal, passing the URL you want to analyze as an argument:

```bash
python main.py <URL>
```

#### **Example**
```bash
python main.py "http://example.com"
```

This command will:
1. Extract URL features.
2. Scrape the website content and run sentiment analysis.
3. Combine the predictions and print the final result to the console.

#### **Output**
The script will output:
- The **final result** indicating whether the URL and its content are likely safe or risky.
- The **sentiment prediction** of the website content (positive or negative).

### **Key Functions**

- **`process_url(url)`**:  
   - Calls the fraud detection system for the provided URL.
   - Outputs the final fraud detection result and sentiment prediction.

- **`fraud_detection(url, folder)`**:  
   - Combines URL feature extraction and sentiment analysis to generate the overall fraud assessment.

- **`ScrapeWebsite(url)`**:  
   - Scrapes website content, extracts contact information, and returns a base folder where the scraped data is stored.

### **Dependencies**

- Python 3.x
- Required libraries:
  - `requests`
  - `BeautifulSoup`
  - `numpy`
  - `joblib`
  - `transformers`
  - `whois`
  - `ipaddress`
  - `re`
  - `socket`
  - `urllib.parse`
  - `tldextract`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

### **File Structure**

- **`main.py`**: The main entry point for processing URLs.
- **`tools/`**: Contains helper scripts for URL fraud detection, sentiment analysis, and web scraping.
- **`result/`**: The folder where scraped website data and logs are saved.


## **2. App Legitimacy Checker via URL**

### **Overview**

This script evaluates the legitimacy of mobile applications based on their Google Play Store listings by analyzing their metadata, reviews, permissions, and scanning for threats using Hybrid Analysis. The script provides a credibility score and suggests whether the app might be fraudulent or trustworthy.

The main script is located in `url_app_check_tool.py`, and it includes functionality to:
- Fetch app metadata from the Google Play Store.
- Analyze reviews and ratings.
- Assess app permissions to identify risky ones.
- Perform a threat intelligence check using Hybrid Analysis.
- Generate an overall credibility score.
- Optionally report apps with a low credibility score as scams.

### **How It Works**

### **1. App Metadata Extraction**
The script fetches the following app details from the Google Play Store:
- App title
- Developer name
- Developer contact email
- App rating
- Number of ratings
- Permissions required by the app

### **2. Review Analysis**
The script retrieves the latest 50 app reviews and calculates:
- The total number of reviews analyzed.
- The count of positive reviews (4 stars and above).
- The count of negative reviews (2 stars and below).

### **3. Permission Analysis**
The script identifies risky permissions from the app's list of permissions. For example, permissions like `READ_SMS`, `SEND_SMS`, `READ_CONTACTS`, and `ACCESS_FINE_LOCATION` are flagged as sensitive and risky.

### **4. Hybrid Analysis Threat Scan**
The app URL is scanned using the Hybrid Analysis API to check for potential malware or other security threats.

### **5. Credibility Score Calculation**
The script generates a credibility score based on several factors:
- App rating and number of ratings.
- Ratio of positive to negative reviews.
- Number of risky permissions.
- Results from the Hybrid Analysis API (e.g., if the app is marked as malicious).

### **6. Reporting Potential Scam Apps**
If the credibility score falls below a certain threshold, the script flags the app as a potential scam and reports it for further investigation.

### **How to Use**

### **Command to Run the Script**

To check an app via its Google Play Store URL, use the following command:

```bash
python url_app_check_tool.py <URL>
```

#### **Example**

```bash
python url_app_check_tool.py "https://play.google.com/store/apps/details?id=com.example.app"
```

#### **Output**

- **App Metadata**:  
   The script will print the app's metadata, including its title, developer, and rating.
  
- **Review Analysis**:  
   It will output the total number of reviews analyzed, the number of positive and negative reviews.

- **Permission Analysis**:  
   The script will print any risky permissions found.

- **Hybrid Analysis Result**:  
   If the app's URL is flagged by the Hybrid Analysis API, the script will display the result.

- **Credibility Score**:  
   A score between 0 and 100 is generated based on the above factors. Lower scores indicate a higher likelihood of the app being fraudulent.

- **Final Verdict**:  
   Based on the credibility score, the app will be classified as:
   - **Potential scam** (if score < 50)
   - **Needs validation** (if score is between 50-60)
   - **Safe to use** (if score > 60)

### **Functions Overview**

- **`process_url(url)`**  
   Processes the app URL by calling the `appCheckViaUrl(url)` function.

- **`get_app_metadata(app_url)`**  
   Fetches metadata from the Google Play Store, such as the app's title, developer, and rating.

- **`analyze_reviews(app_url)`**  
   Analyzes the most recent 50 reviews to determine the number of positive and negative reviews.

- **`analyze_permissions(permissions)`**  
   Checks the app's permissions for risky ones like `READ_SMS` or `ACCESS_FINE_LOCATION`.

- **`check_with_hybrid_analysis(app_url)`**  
   Sends the app URL to the Hybrid Analysis API to check for potential threats.

- **`generate_credibility_score(metadata, review_analysis, permission_analysis, hybrid_analysis)`**  
   Generates a credibility score based on app metadata, reviews, permissions, and Hybrid Analysis results.

- **`appCheckViaUrl(app_url)`**  
   Combines all the above functions to provide a final assessment of the app's legitimacy.

### **Dependencies**

- Python 3.x
- Required libraries:
  - `google-play-scraper`
  - `requests`
  - `json`

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

### **Additional Notes**

1. **Hybrid Analysis API**:  
   Ensure you have a valid Hybrid Analysis API key and replace the placeholder `api_key` with your actual key in the `check_with_hybrid_analysis` function.

2. **Usage Warnings**:  
   This tool is for educational purposes and testing. Always follow legal guidelines when using APIs or accessing app data.

## **Conclusion**

This script provides an effective way to verify the legitimacy of apps on the Google Play Store by combining metadata analysis, review and permission checks, and threat intelligence scanning.


## **3. Image Sentiment Analysis and Fraud Detection**

### **Overview**

This Python script analyzes the sentiment of text extracted from images using Optical Character Recognition (OCR) and a machine learning sentiment analysis model. It is designed to detect whether the content in an image potentially indicates fraud or not.

The main script is located in `main.py`, and it performs the following steps:
1. Extract text from an image using OCR.
2. Analyze the sentiment of the extracted text.
3. Provide a verdict on whether the content could be indicative of fraud based on the sentiment.

### **How It Works**

### **1. Optical Character Recognition (OCR)**
The script uses EasyOCR to extract text from the provided image. It reads the text content of the image and can display the image with bounding boxes around the detected text for visual confirmation.

### **2. Sentiment Analysis**
The sentiment analysis is performed using a pre-trained machine learning model. The extracted text is preprocessed, vectorized, and passed through the sentiment model to predict whether the text is "positive" (potential fraud) or "negative" (not fraud).

### **3. Fraud Detection**
- If the sentiment is positive, the content may indicate **potential fraud**.
- If the sentiment is negative, the content is considered **not potential fraud**.

### **How to Use**

### **Command to Run the Script**

To analyze the sentiment of text in an image, use the following command:

```bash
python main.py <IMAGE_PATH>
```

#### **Example**

```bash
python main.py "path/to/image.jpg"
```

#### **Output**

- The script will print the predicted sentiment (either "positive" or "negative").
- Based on the sentiment, it will print:
  - **"Potential Fraud"** if the sentiment is positive.
  - **"Not potential Fraud"** if the sentiment is negative.
- If no text is extracted from the image, it will output **"No text extracted from the image."**

### **Functions Overview**

- **`process_path(path)`**:  
   Processes the image at the given path and runs the sentiment analysis.

- **`load_sentiment_model()`**:  
   Loads the pre-trained sentiment analysis model and vectorizer from saved files.

- **`PreProcessText(review)`**:  
   Preprocesses the extracted text by converting it to lowercase.

- **`predict_sentiment(review, grid_search, vectorizer)`**:  
   Vectorizes the preprocessed text and predicts its sentiment using the loaded model.

- **`ocr_with_easyocr(image_path)`**:  
   Uses EasyOCR to extract text from the given image. Optionally displays the image with the recognized text.

- **`analyze_sentiment(image_path)`**:  
   Combines OCR and sentiment analysis to predict whether the content in the image suggests potential fraud.

### **Dependencies**

- Python 3.x
- Required libraries:
  - `easyocr`
  - `opencv-python`
  - `joblib`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### **Model Files**

The sentiment analysis model is saved in the following files:
- **`Limitedsentiment.pkl`**: The trained sentiment analysis model.
- **`Limitedvectorizer.pkl`**: The vectorizer used to preprocess the text data before sentiment prediction.

Ensure these files are present in the `tools` directory.

### **File Structure**

- **`main.py`**: The main entry point for processing images and detecting potential fraud.
- **`tools/`**: Contains helper functions for sentiment analysis and OCR processing.

### **Conclusion**

This script provides a simple yet effective way to analyze text in images and determine whether the content could be related to potential fraud using OCR and sentiment analysis.


## **4. Contact Verification Tool for Fraud Detection**

### **Overview**

This Python script is designed to verify phone numbers against a known fraud database. It checks if the provided phone number exists in a verified fraud database, flags it if found, or adds it to a suspect database for further investigation. This tool is useful for detecting potentially fraudulent customer care numbers and other contact details.

### **How It Works**

1. **Phone Number Verification:**
   - The tool cross-references a provided phone number with a verified fraud database (`verified_data.csv`).
   - If the phone number is found in the fraud database, it raises an alert, displaying the suspect's name.
   - If the phone number is not found, it adds the number to a suspect database (`suspect_data.csv`) for further investigation.

2. **CSV File Operations:**
   - **`verified_data.csv`**: Contains verified fraud records with fields like suspect name and phone number.
   - **`suspect_data.csv`**: Stores newly identified suspect phone numbers that were not found in the verified database, marking them for future investigation.

### **Key Features**

- **Fraud Detection**:
   The tool checks if the provided phone number exists in the verified fraud database. If a match is found, it provides an alert with the suspect's name.

- **Suspect Database**:
   If the phone number is not found in the fraud database, the tool automatically adds it to a suspect database for further scrutiny, ensuring continuous monitoring and updating.

- **Phone Number Normalization**:
   The tool processes phone numbers by normalizing them (removing non-numeric characters) to ensure consistency in matching.

### **Usage**

#### **Command to Run the Script**

To verify a phone number, run the following command:

```bash
python verify_contacts_tool.py <PHONE_NUMBER>
```

#### **Example**

```bash
python verify_contacts_tool.py 1234567890
```

#### **Output**

- If the phone number is found in the fraud database:
  ```bash
  🚨 ALERT: The phone number <PHONE_NUMBER> is a confirmed fraud and found in the fraud database! Suspect name: <SUSPECT_NAME>
  ```
  
- If the phone number is not found:
  ```bash
  The phone number <PHONE_NUMBER> is not in the fraud database. Adding to the suspect database.
  ✅ The phone number <PHONE_NUMBER> has been added to the suspect database for further investigation.
  ```

- If the phone number format is invalid:
  ```bash
  Invalid input. Please enter a valid float for the phone number.
  ```

### **Functions Overview**

- **`process_phone(phone)`**:  
   Takes a phone number as input, calls the `search_verified_contact` function, and displays whether the number is fraudulent or will be added to the suspect database.

- **`search_verified_contact(number)`**:  
   Cross-checks the given phone number with the verified fraud database. If not found, it adds the number to the suspect database.

- **`search(file_path, id_number)`**:  
   Searches the CSV file for a matching phone number. If found, it returns the suspect's details.

- **`addToSuspectDatabase(new_number, notes)`**:  
   Adds a new suspect phone number to `suspect_data.csv` for further investigation.

- **`normalize_phone_number(phone_number)`**:  
   Removes all non-numeric characters from the phone number to ensure consistency in matching.

### **Files**

- **`verified_data.csv`**:  
   This file contains the verified fraud database with the following columns:
   - `suspectname`: Name of the fraud suspect.
   - `idnumber`: Verified fraud phone number.

- **`suspect_data.csv`**:  
   This file contains phone numbers that were not found in the fraud database but have been added to the suspect list for further investigation. The columns include:
   - `suspectname`: Name of the suspect (initially set to 'Unknown').
   - `idnumber`: Phone number added to the suspect list.
   - `notes`: Any additional notes about the suspect.

### **Dependencies**

- Python 3.x
- Required libraries:
  - `pandas`
  - `re`

Install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

### **Considerations**

- The tool currently processes phone numbers as float inputs. Be cautious when entering phone numbers, as they should be in valid float format (without special characters like `+` or `-`).
- The system automatically updates the `suspect_data.csv` file when a phone number is not found in the verified database. Ensure you have write permissions for this file's location.

### **Example Scenario**

If you need to check a customer care phone number for a possible fraud, simply run the script with the phone number as an argument. The tool will alert you if the phone number is confirmed as fraudulent. If the phone number is not yet flagged but suspicious, the tool will automatically add it to the suspect database for future investigation.

### **Conclusion**

This tool is useful for cross-referencing phone numbers and contact details against a verified fraud database. It adds new suspect numbers to a database for continuous monitoring and investigation, ensuring a proactive approach to fraud detection.



# **5. APK Analysis Tool for Fraud Detection**

### **Overview**

This tool performs comprehensive security analysis on Android APK files to detect potentially malicious applications. The script uses various techniques such as VirusTotal lookup, permission analysis, obfuscation detection, APK signature verification, and Google Play Store metadata extraction. The results are combined to make a final decision on whether the APK is potentially malicious or legitimate.

### **Features**

1. **Google Play Store Lookup**  
   Checks if the APK's package name is listed on the Google Play Store and retrieves app metadata such as title, developer, and ratings.

2. **VirusTotal API Scan**  
   Queries VirusTotal for any existing scan reports for the APK, or uploads the APK to VirusTotal for a new scan if no report is found. Reports any positive detections by VirusTotal.

3. **Permission Analysis**  
   Analyzes the APK for permissions and highlights potentially dangerous ones (e.g., `READ_SMS`, `SEND_SMS`, `ACCESS_FINE_LOCATION`, etc.).

4. **Obfuscation Detection**  
   Uses two methods to detect obfuscation in APKs:
   - **APKiD**: Detects obfuscation by checking APK attributes.
   - **Androguard**: Detects obfuscation through class names and method analysis.

5. **APK Signature Verification**  
   Verifies the APK signature using RSA and checks for any inconsistencies in the APK's signing certificate.

6. **APK Hash Calculation**  
   Computes the SHA-256 hash of the APK for integrity checks.

7. **APK Extraction**  
   Extracts APK resources for manual inspection (optional functionality for future use).

### **Usage**

#### **Command to Run the Script**

To analyze an APK file, use the following command:

```bash
python apk_tool.py <APK Path>
```

#### **Example**

```bash
python apk_tool.py "path/to/app.apk"
```

#### **Output**

- **APK Metadata**:  
  - Package Name
  - Version Name
  - Version Code

- **Hash Value**:  
  The SHA-256 hash of the APK file.

- **Permission Analysis**:  
  - List of all permissions requested by the app.
  - Highlights any suspicious permissions.

- **VirusTotal Results**:  
  - Displays the number of positive detections, if any.
  - Uploads the APK to VirusTotal for a scan if no prior report is found.

- **Obfuscation Detection**:  
  - Reports whether the APK is obfuscated using APKiD and Androguard.

- **Signature Verification**:  
  - Checks if the APK signature is valid or has any issues.

- **Final Decision**:  
  Based on the findings, the tool outputs whether the APK is potentially malicious or legitimate.

### **Functions Overview**

- **`detect_malicious_apk(apk_file_path, api_keys)`**:  
   The main function that combines all analysis steps and outputs the final decision.
   
- **`check_google_play_store(package_name)`**:  
   Checks whether the app's package is available on the Google Play Store and fetches its metadata.

- **`check_virustotal(apk_file_path, api_key)`**:  
   Queries VirusTotal for a report on the APK or uploads the APK for a new scan.

- **`analyze_permissions(apk_file_path)`**:  
   Analyzes the APK's permissions, highlighting suspicious permissions.

- **`check_obfuscation_apkid(apk_file_path)`**:  
   Uses APKiD to detect obfuscation techniques in the APK.

- **`detect_obfuscation_androguard(apk_file_path)`**:  
   Uses Androguard to detect obfuscation based on class names and methods.

- **`verify_apk_signature(apk_file_path)`**:  
   Verifies the APK's signature using RSA and checks for any inconsistencies.

- **`get_file_hash(file_path, algorithm='sha256')`**:  
   Calculates the SHA-256 hash of the APK for integrity checks.

### **Dependencies**

- Python 3.x
- Required libraries:
  - `requests`
  - `androguard`
  - `google-play-scraper`
  - `pycryptodome`
  - `APKiD`
  - `hashlib`
  - `zipfile`
  - `subprocess`
  
Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

### **API Keys**

The tool uses the **VirusTotal API** for scanning APK files. You will need to provide your VirusTotal API key for the script to function properly.

You can replace the placeholder API key in the code with your own key:

```python
api_keys = {"virustotal": "your_virustotal_api_key"}
```

### **File Structure**

- **`apk_tool.py`**: Main script for analyzing APKs.
- **`tools/`**: Directory containing helper functions (if necessary for modular use).

### **Example Workflow**

1. **Run the script with the APK path** to analyze an APK.
2. **The tool performs the following checks**:
   - Google Play Store Lookup
   - VirusTotal Scan
   - Permission Analysis
   - Obfuscation Detection (APKiD and Androguard)
   - Signature Verification
3. **Based on the findings**, the tool will indicate whether the APK is likely malicious or legitimate.




#### Team details
    Team Name:    Processing2o
    University:   NFSU, Goa
    Team members: Haardik Paras Bhagtani
                  Mihir Ranjan
                  Abhijeet Kumar
                  Swarnim Jodh