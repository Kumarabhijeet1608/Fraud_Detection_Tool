import sys
from tools.url_detection_tool import fraud_detection
from tools.scraper import ScrapeWebsite

def process_url(url):
    print(f"Processing URL: {url}")
    result = fraud_detection(url, ScrapeWebsite(url)['base_folder'])
    print('Final result: ', result['final_result'])
    print('Sentiment Prediction: ', result['sentiment_prediction'])

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    process_url(url)

if __name__ == "__main__":
    main()