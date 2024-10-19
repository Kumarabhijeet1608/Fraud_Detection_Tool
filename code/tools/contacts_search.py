import pandas as pd
import re

# Load the existing suspect_data.csv file
def load_verified_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        return None

# Function to normalize phone numbers by removing non-digit characters
def normalize_phone_number(phone_number):
    if pd.isna(phone_number):  # Check if the phone number is NaN
        return ''
    return re.sub(r'\D', '', str(phone_number))  # Remove all non-numeric characters

# Function to convert float input into a valid phone number float
def convert_float_to_phone_string(float_number):
    # Convert the float to an integer (this removes the decimal point)
    int_number = int(float_number)
    # Convert the integer back to a float
    phone_float = float(int_number)
    
    return phone_float

# Function to validate and update CSV based on user input
def search(file_path, id_number=None):
    # Load the existing suspect data
    df = load_verified_data(file_path)
    
    if df is None:
        return "Error: Verified data file not found."
    # Normalize the phone numbers in the CSV and the provided input
    if id_number:
        id_number_normalized = normalize_phone_number(id_number)
        df['idnumber_normalized'] = df['idnumber'].apply(normalize_phone_number)

    # Check for phone number matches in the CSV (exact match for normalized numbers)
    if id_number:
        id_match = df[df['idnumber_normalized'] == id_number_normalized]
        if not id_match.empty:
            return {
                'exists': True,
                'suspect': id_match.iloc[0]['suspectname']
            }
        else:
            addToSuspectDatabase(id_number_normalized, '')
            return {
                'exists': False,
                'suspect': 'Sespect Added to Suspect database'
            }        

# Function to take input from the user and call the validation function
def search_verified_contact(number: int):
    """
        If the tool exists in verified fraud database, it will alert us, or else it will add the number in suspect database
    """
    verified_data_path = 'tools\\verified_data.csv'
    
    try:
        # Take the phone number as a float input
        # id_number = convert_float_to_phone_string(number)
        result = search(verified_data_path, id_number=float(number))
    except ValueError:
        result = "Invalid input. Please enter a valid float for the phone number."
    
    return result

def addToSuspectDatabase(new_number, notes):
     # Append the new record to updated_suspect_data.csv
    new_record = pd.DataFrame({
        'suspectname': ['Unknown'],  # You can modify this if you want to prompt for a name
        'idnumber': [new_number],
        'notes': [notes]
    })
    
    new_file_path = 'tools\\suspect_data.csv'
    
    try:
        new_record.to_csv(new_file_path, mode='a', index=False, header=False)
        return f"New record added."
    except Exception as e:
        return f"Error updating the CSV file: {str(e)}"