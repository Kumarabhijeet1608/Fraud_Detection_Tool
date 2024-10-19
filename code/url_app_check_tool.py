import sys
from tools.app_check_by_url import appCheckViaUrl

def process_url(url):
    print(f"Processing URL: {url}")
    appCheckViaUrl(url)



def main():
    if len(sys.argv) != 2:
        print("Usage: python url_app_check_tool.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    process_url(url)

if __name__ == "__main__":
    main()