import sys
from tools.image_sentiments_detect import analyze_sentiment

def process_path(path):
    print(f"Processing Image Path: {path}")
    analyze_sentiment(path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <PATH>")
        sys.exit(1)
    
    path = sys.argv[1]
    process_path(path)

if __name__ == "__main__":
    main()