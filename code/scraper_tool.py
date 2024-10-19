from tools.scraper import ScrapeWebsite
import sys

def process_url(url):
    print(f"Processing URL: {url}")

    scraped_result_data = ScrapeWebsite(url)
    print(f"\n🚀 Scraping Complete! Your data has been successfully captured and saved to \\result\\{scraped_result_data['base_folder']}\\ folder! 📂")
    print("🔍 Dive into the details: text, images, and contacts are all organized and ready! 🗂️")
    print("⏱️ Average time per page: {:.2f} seconds. Efficiency at its finest! ⚡".format(scraped_result_data['avg_time']))
    print("🌐 All done! Ready for the next URL adventure? 🕵️‍♂️💻")



def main():
    if len(sys.argv) != 2:
        print("Usage: python scraper_tool.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    process_url(url)

if __name__ == "__main__":
    main()