import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from collections import Counter
from string import punctuation
import time
import tldextract
from verify_contacts_tool import process_phone

# Function to create the base result folder
def create_folders(base_folder="result"):
    Path(base_folder).mkdir(parents=True, exist_ok=True)

# Function to log forbidden/broken websites for each respective site folder
def log_forbidden_link(url, reason, forbidden_file):
    with open(forbidden_file, 'a', encoding='utf-8') as file:
        file.write(f"{url} - {reason}\n")

# Function to check if a link is internal (i.e., belongs to the same domain)
def is_internal_link(base_url, link):
    parsed_base_url = urlparse(base_url)
    parsed_link = urlparse(link)
    return (parsed_link.netloc == "" or parsed_base_url.netloc == parsed_link.netloc)

# Function to download images into the appropriate folder
def download_image(session, img_url, img_folder):
    try:
        img_data = session.get(img_url).content
        img_name = os.path.basename(urlparse(img_url).path)
        if img_name:
            img_path = os.path.join(img_folder, img_name)
            with open(img_path, 'wb') as img_file:
                img_file.write(img_data)
            print(f"Image saved: {img_path}")
    except Exception as e:
        print(f"Failed to download image {img_url}: {e}")

# Function to scrape additional pages like robots.txt, sitemap.xml, etc.
def scrape_additional_pages(session, base_url):
    additional_pages = ['/robots.txt', '/sitemap.xml', '/rss.xml', '/feed.xml', '/humans.txt', '/server-status', '/admin', '/login']
    found_links = set()
    
    for page in additional_pages:
        page_url = urljoin(base_url, page)
        try:
            response = session.get(page_url)
            if response.status_code == 200:
                print(f"Found {page}: {page_url}")
                # Special handling for XML files like sitemap and RSS
                if page.endswith('.xml'):
                    soup = BeautifulSoup(response.content, 'xml')
                    urls = [loc.get_text() for loc in soup.find_all('loc')]  # Only valid for sitemaps
                    found_links.update(urls)
                else:
                    found_links.add(page_url)
        except Exception as e:
            print(f"Could not retrieve {page_url}: {e}")
    return found_links

# Function to count most common words in a webpage
def count_common_words(soup):
    # Extract text from paragraphs and divs
    text_paragraph = ' '.join(s.get_text() for s in soup.find_all('p'))
    text_div = ' '.join(s.get_text() for s in soup.find_all('div'))
    
    # Combine the two
    combined_text = f"{text_paragraph} {text_div}"
    
    # Count words
    count = Counter((x.rstrip(punctuation).lower() for x in combined_text.split()))
    return count


# Function to extract contact information (phone numbers, emails, addresses)
def extract_contact_info(soup):
    text = soup.get_text()
    phone_regex = r"""
        (?:
            # Country code in the format +XX or (XXX) or XX-
            (?:\+?\d{1,3}[-.\s]?)?
            # Optional area code or space between country and local number
            (?:\(?\d{1,4}\)?[-.\s]?)?
            # First set of digits (e.g., first part of phone number)
            (?:\d{1,5}[-.\s]?)
            # Second set of digits (e.g., second part of phone number)
            (?:\d{1,5}[-.\s]?)
            # Remaining digits
            (?:\d{1,9})
        )
        """

    phones = [format_phone_number(num) for num in re.findall(phone_regex, text, re.VERBOSE)]
    try:
        for phone in phones:
            if(phone == "0"):
                return
            process_phone(int(phone))
    except requests.RequestException as e:
        print(e)
                
    return {"phones": phones}

def format_phone_number(phone_number):
    # Remove spaces, special characters, etc., by keeping only digits
    return re.sub(r'\D', '', phone_number)  # \D matches any non-digit character

def extract_email_info(soup):
    text = soup.get_text()
    email_regex = r'[\w\.-]+@[\w\.-]+'
    emails = re.findall(email_regex, text)
    return {"emails": emails}

def extract_address_info(soup):
    text = soup.get_text()
    address_keywords = r'(street|st\.|road|rd\.|avenue|ave\.|block|area|lane|city|town|state|zip|pincode|sector|building|floor)'
    addresses = re.findall(rf'(\b.*?\b(?:{address_keywords}).*?\b)', text, re.IGNORECASE)
    return {"addresses": addresses}

# Function to save contact details into a contacts.txt file
def save_contacts_to_file(contacts, emails, addresses, contacts_file_name, emails_file_name, address_file_name):
    with open(contacts_file_name, 'w', encoding='utf-8') as file:
        file.writelines(f"{phone}\n" for phone in contacts["phones"])
    with open(emails_file_name, 'w', encoding='utf-8') as file:
        file.writelines(f"{email}\n" for email in emails["emails"])
    with open(address_file_name, 'w', encoding='utf-8') as file:
        file.writelines(f"{address}\n" for address in addresses["addresses"])

# Main function to scrape URL content
def scrape_url_content(session, url, visited, base_url, text_file_name, images_folder, count_file_name, contacts_file_name, emails_file_name, address_file_name, forbidden_file):
    if url in visited:
        return
    visited.add(url)

    try:
        start_time = time.time()
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            log_forbidden_link(url, f"HTTPError: {response.status_code}", forbidden_file)
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text()

        # Append scraped content to a single file
        save_to_file(content, text_file_name)

        # Download all images from the current page
        for img_tag in soup.find_all('img', src=True):
            img_url = urljoin(base_url, img_tag['src'])
            download_image(session, img_url, images_folder)

        # Count and save the most common words
        common_words = count_common_words(soup)
        save_word_count_to_file(common_words, count_file_name)

        # Extract and save contact information
        contacts = extract_contact_info(soup)
        emails = extract_email_info(soup)
        addresses = extract_address_info(soup)
        save_contacts_to_file(contacts, emails, addresses, contacts_file_name, emails_file_name, address_file_name)

        # Recursively follow all internal links
        for link in soup.find_all('a', href=True):
            full_url = urljoin(base_url, link['href'])
            if is_internal_link(base_url, full_url):
                scrape_url_content(session, full_url, visited, base_url, text_file_name, images_folder, count_file_name, contacts_file_name, emails_file_name, address_file_name, forbidden_file)

        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time

    except requests.RequestException as e:
        print(e)
        log_forbidden_link(url, str(e), forbidden_file)

def save_to_file(content, file_name):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(content)
        file.write("\n\n" + "-"*80 + "\n\n")

def save_word_count_to_file(counter, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for word, count in counter.most_common():
            file.write(f"{word}: {count}\n")

def extract_base_domain(url):
    extract = tldextract.extract(url)
    # Create folder name like "example.com"
    return f"{extract.domain}.{extract.suffix}"

# Main function to process multiple URLs
def ScrapeWebsite(url: str):
    """
        This script recursively scrapes websites, extracting and saving text, images, word counts, and contact information (phones, emails, addresses) into organized directories. It uses an HTTP session for efficient requests, follows internal links, and logs errors or forbidden URLs. The tool tracks and returns the average time taken to scrape each page.
    """
    visited = set()
    create_folders()

    total_time = 0
    count = 0

    try:
        base_domain = extract_base_domain(url)
        folder_name = base_domain + str(time.time())
        website_folder = os.path.join("result", folder_name)
        text_file_name = os.path.join(website_folder, "scrape.txt")
        images_folder = os.path.join(website_folder, "images")
        count_file_name = os.path.join(website_folder, "counts_words.txt")
        contacts_file_name = os.path.join(website_folder, "contacts.txt")
        emails_file_name = os.path.join(website_folder, "emails.txt")
        address_file_name = os.path.join(website_folder, "address.txt")
        forbidden_file = os.path.join(website_folder, "forbidden_links.txt")

        Path(images_folder).mkdir(parents=True, exist_ok=True)

        with requests.Session() as session:
            elapsed_time = scrape_url_content(session, url, visited, url, text_file_name, images_folder, count_file_name, contacts_file_name, emails_file_name, address_file_name, forbidden_file)
            if elapsed_time:
                total_time += elapsed_time
                count += 1

    except Exception as e:
        return {"error": f"Failed to scrape {url}: {e}"}

    avg_time = total_time / count if count > 0 else 0
    return {'base_folder': folder_name, 'avg_time': avg_time}

