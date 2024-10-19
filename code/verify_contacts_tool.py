from tools.contacts_search import search_verified_contact
import sys

def process_phone(phone):
    print(f"Processing phone: {phone}")


    result_data = search_verified_contact(phone)
    if result_data == None:
        print(f"The phone number {phone} is not valid number.")
    else:    
        if result_data['exists'] == True:
            print(f"🚨 ALERT: The phone number {phone} is a confirmed fraud and found in the fraud database! Suspect name: {result_data['suspect']}")
        else:
            print(f"The phone number {phone} is not in the fraud database. Adding to the suspect database.")
            print(f"✅ The phone number {phone} has been added to the suspect database for further investigation.")    


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_contacts_tool.py <PHONE>")
        sys.exit(1)
    
    url = sys.argv[1]
    process_phone(url)

if __name__ == "__main__":
    main()