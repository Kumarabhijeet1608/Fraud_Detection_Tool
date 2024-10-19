from google_play_scraper import app, reviews, Sort
import requests

# Function to get metadata from Google Play Store
def get_app_metadata(app_url):
    try:
        app_id = app_url.split('id=')[1]
        app_data = app(app_id)

        metadata = {
            'title': app_data['title'],
            'developer': app_data['developer'],
            'developer_email': app_data.get('developerEmail'),
            'rating': app_data['score'],
            'ratings_count': app_data['ratings'],
            'permissions': app_data.get('permissions', 'Not Available'),
            'url': app_url,
        }
        return metadata
    except Exception as e:
        return {'error': str(e)}

# Function to analyze reviews and ratings
def analyze_reviews(app_url):
    try:
        app_id = app_url.split('id=')[1]
        result, _ = reviews(
            app_id,
            lang='en',  # Reviews in English
            country='us',  # United States
            sort=Sort.NEWEST,  # Sort by newest reviews
            count=50  # Fetch 50 reviews for analysis
        )

        positive_reviews = sum(1 for r in result if r['score'] >= 4)
        negative_reviews = sum(1 for r in result if r['score'] <= 2)

        review_analysis = {
            'total_reviews': len(result),
            'positive_reviews': positive_reviews,
            'negative_reviews': negative_reviews,
        }
        return review_analysis
    except Exception as e:
        return {'error': str(e)}

# Function to analyze app permissions
def analyze_permissions(permissions):
    try:
        sensitive_permissions = ['READ_SMS', 'SEND_SMS', 'READ_CONTACTS', 'ACCESS_FINE_LOCATION']
        risky_permissions = [p for p in sensitive_permissions if p in permissions]
        permission_analysis = {
            'risky_permissions': risky_permissions,
            'risky_count': len(risky_permissions),
        }
        return permission_analysis
    except Exception as e:
        return {'error': str(e)}


# Function to check the app URL with Hybrid Analysis for threat intelligence
def check_with_hybrid_analysis(app_url):
    try:
        api_key = 'bco2xe3h9118230bqj26tpej739ae3bbifny72do68cc5c8398sxx2s8322af9fe'  # Replace with your Hybrid Analysis API key
        headers = {
            'api-key': api_key,
            'user-agent': 'Falcon Sandbox',
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {
            'url': app_url,  # Ensure that the URL is being passed correctly
            'scan_type': 'all',  # Use a valid scan type as per API documentation, e.g., "quick" or "full"
        }
        # Hybrid Analysis URL for URL scanning
        url = 'https://www.hybrid-analysis.com/api/v2/quick-scan/url'

        # Send the request
        response = requests.post(url, data=data, headers=headers)

        # Check if the response was successful
        if response.status_code == 200:
            return response.json()  # Return the JSON response if successful
        else:
            return {'error': response.status_code, 'message': response.text}
    
    except Exception as e:
        return {'error': str(e)}


# Function to generate a credibility score based on collected data
def generate_credibility_score(metadata, review_analysis, permission_analysis, hybrid_analysis):
    try:
        score = 100  # Start with 100 and deduct based on issues

        # Check if rating exists, if not assign 0 or handle accordingly
        app_rating = metadata['rating'] if metadata['rating'] is not None else 0
        ratings_count = metadata['ratings_count'] if metadata['ratings_count'] is not None else 0

        if app_rating < 3.5 or ratings_count < 100:
            score -= 30
        if review_analysis['negative_reviews'] > review_analysis['positive_reviews']:
            score -= 20
        if permission_analysis['risky_count'] > 2:
            score -= 20
        if 'malicious' in hybrid_analysis.get('verdict', ''):
            score -= 40

        return score  # Return the score directly
    except Exception as e:
        return {'error': str(e)}  # Handle errors by returning a dictionary

# Function to report a false app if identified as a scam
def report_false_app(app_url):
    # Add feedback mechanism here, such as emailing, logging, or sending to a reporting API.
    print(f"Reported {app_url} as a potential scam.")

# Main function to execute the entire process
def appCheckViaUrl(app_url):
    print(f"Checking app: {app_url}")

    # Step 1: Fetch metadata
    metadata = get_app_metadata(app_url)
    if 'error' in metadata:
        print(f"Error in fetching metadata: {metadata['error']}")
        return

    # Step 2: Analyze reviews
    review_analysis = analyze_reviews(app_url)
    if 'error' in review_analysis:
        print(f"Error in analyzing reviews: {review_analysis['error']}")
        return

    # Step 3: Analyze permissions
    permission_analysis = analyze_permissions(metadata['permissions'])
    if 'error' in permission_analysis:
        print(f"Error in analyzing permissions: {permission_analysis['error']}")
        return

    # Step 5: Check Hybrid Analysis API
    hybrid_analysis = check_with_hybrid_analysis(app_url)
    if 'error' in hybrid_analysis:
        print(f"Error in Hybrid Analysis: {hybrid_analysis['message']}")
        return

    # Step 6: Generate credibility score
    score = generate_credibility_score(metadata, review_analysis, permission_analysis,hybrid_analysis)

    # Output Results
    print(f"App: {metadata['title']}")
    print(f"Developer: {metadata['developer']}")
    print(f"Rating: {metadata['rating']}")
    print(f"Total Reviews: {review_analysis['total_reviews']}")
    print(f"Positive Reviews: {review_analysis['positive_reviews']}")
    print(f"Negative Reviews: {review_analysis['negative_reviews']}")
    print(f"Permissions: {metadata['permissions']}")
    print(f"Credibility Score: {score}")

    # Step 7: Report app if score is low
    if score < 50:
        print(f"Reported {app_url} as a potential scam.")
    elif score>=50 and score<=60:
        print(f"Reported {app_url} can be a potential scam. Please validate!")
    elif score>60:
        print(f"Reported {app_url} is good to go!")

