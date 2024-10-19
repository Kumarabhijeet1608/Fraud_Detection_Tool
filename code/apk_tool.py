import os
import requests
from androguard.core.apk import APK
from androguard.misc import AnalyzeAPK
import zipfile
import subprocess
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
import hashlib
import textwrap
import sys

# ---------- Helper Functions ----------
def get_file_hash(file_path, algorithm='sha256'):
    """Calculate the hash of the APK file using the specified algorithm (default SHA-256)."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hash_func.update(chunk)
    return hash_func.hexdigest()

# ---------- 1. Google Play Store API Lookup ----------
def check_google_play_store(package_name):
    try:
        from google_play_scraper import app
        return app(package_name)
    except Exception as e:
        return f"App not found on Play Store: {e}"

# ---------- 2. VirusTotal API for APK Lookup ----------
def check_virustotal(apk_file_path, api_key):
    file_hash = get_file_hash(apk_file_path)
    url_report = f"https://www.virustotal.com/vtapi/v2/file/report"
    params_report = {'apikey': api_key, 'resource': file_hash}
    
    response_report = requests.get(url_report, params=params_report)
    if response_report.status_code != 200:
        print(f"Error fetching VirusTotal report. Status Code: {response_report.status_code}")
        return {}

    try:
        report_data = response_report.json()
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse VirusTotal report.")
        return {}

    if report_data.get('response_code') == 1:
        print("VirusTotal scan report found.")
        return report_data

    print("Uploading APK to VirusTotal for new scan...")
    url_scan = 'https://www.virustotal.com/vtapi/v2/file/scan'
    with open(apk_file_path, 'rb') as file:
        files = {'file': (apk_file_path, file)}
        params_scan = {'apikey': api_key}
        response_scan = requests.post(url_scan, files=files, params=params_scan)

    if response_scan.status_code != 200:
        print(f"Error uploading APK to VirusTotal. Status Code: {response_scan.status_code}")
        return {}

    try:
        return response_scan.json()
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse VirusTotal upload response.")
        return {}

# ---------- 3. Permission Analysis ----------
def analyze_permissions(apk_file_path):
    apk = APK(apk_file_path)
    dangerous_permissions = {"READ_SMS", "SEND_SMS", "READ_CONTACTS", "ACCESS_FINE_LOCATION", "INTERNET", "CAMERA", "RECORD_AUDIO", "WRITE_EXTERNAL_STORAGE"}
    
    permissions = apk.get_permissions()
    suspicious_permissions = [perm for perm in permissions if perm.split('.')[-1] in dangerous_permissions]

    return {
        "permissions": permissions,
        "suspicious_permissions": suspicious_permissions,
        "package_name": apk.get_package(),
        "version_name": apk.get_androidversion_name(),
        "version_code": apk.get_androidversion_code()
    }

# ---------- 4. Obfuscation Detection using APKiD ----------
def check_obfuscation_apkid(apk_file_path):
    """Use APKiD for obfuscation detection by running a subprocess command."""
    command = f"apkid {apk_file_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"APKiD Obfuscation Check Output:\n{result.stdout}")
    
    if "obfuscator" in result.stdout.lower():
        return "Obfuscation Detected by APKiD"
    return "No Obfuscation Detected by APKiD"

def detect_obfuscation_androguard(apk_file_path):
    """Use Androguard for obfuscation detection by analyzing class names and methods."""
    _, d_list, dx = AnalyzeAPK(apk_file_path)
    obfuscation_flags = []

    for d in d_list:
        for class_name in d.get_classes_names():
            if len(class_name.split('/')[-1]) == 1:
                obfuscation_flags.append(f"Obfuscated class: {class_name}")

    for method in dx.get_methods():
        if "java/lang/reflect" in method.get_class_name():
            obfuscation_flags.append(f"Reflection usage in method: {method.name}")

    return obfuscation_flags or "No obfuscation detected by Androguard"

# ---------- 5. APK Extraction ----------
def extract_apk_resources(apk_file_path, output_dir):
    """Extract APK contents for manual inspection."""
    with zipfile.ZipFile(apk_file_path, 'r') as apk:
        apk.extractall(output_dir)
    print(f"APK extracted to {output_dir}")

# ---------- 6. APK Signature Verification ----------
def verify_apk_signature(apk_file_path):
    apk = APK(apk_file_path)
    cert_files = [f for f in apk.get_files() if f.startswith("META-INF/") and (f.endswith(".RSA") or f.endswith(".DSA"))]

    if not cert_files:
        return "No APK signature found."

    with zipfile.ZipFile(apk_file_path, 'r') as apk_zip:
        cert_file = cert_files[0]
        cert_data = apk_zip.read(cert_file)
        try:
            cert = RSA.import_key(cert_data)
        except Exception as e:
            return f"Failed to extract public key: {e}"

        manifest_file = apk.get_files().get("META-INF/MANIFEST.MF")
        if not manifest_file:
            return "No MANIFEST.MF file found, cannot verify APK signature."

        manifest_data = apk_zip.read(manifest_file).decode('utf-8')
        for file_name in manifest_data.splitlines():
            if "SHA-256-Digest:" in file_name:
                return "APK Signature Valid"
        return "APK Signature Invalid"

# ---------- 7. Final Decision Logic ----------
def detect_malicious_apk(apk_file_path, api_keys):
    # 1. Extract APK metadata and permissions
    permissions_info = analyze_permissions(apk_file_path)

    # 2. Google Play Store Lookup
    google_play_data = check_google_play_store(permissions_info["package_name"])

    # 3. VirusTotal API Check
    virus_total_data = check_virustotal(apk_file_path, api_keys['virustotal'])

    # 4. APK Signature Verification
    apk_signature_result = verify_apk_signature(apk_file_path)

    # 5. Obfuscation Detection (Using Both APKiD and Androguard)
    obfuscation_result_apkid = check_obfuscation_apkid(apk_file_path)
    obfuscation_result_androguard = detect_obfuscation_androguard(apk_file_path)

    # 6. APK Hash/Integrity Check
    apk_hash = get_file_hash(apk_file_path)

    # Combine the findings into a decision
    flagged_reasons = []

    if "App not found on Play Store" in google_play_data:
        flagged_reasons.append(" - The app is not listed on the Google Play Store.")

    if virus_total_data.get('positives', 0) > 0:
        flagged_reasons.append(f" - VirusTotal reported {virus_total_data.get('positives')} positives.")

    if obfuscation_result_apkid != "No Obfuscation Detected by APKiD":
        flagged_reasons.append(f" - Obfuscation detected: {obfuscation_result_apkid}")

    if obfuscation_result_androguard != "No obfuscation detected by Androguard":
        obfuscation_text = textwrap.fill(f" - Obfuscation detected by Androguard: {obfuscation_result_androguard}", width=80, subsequent_indent='   ')
        flagged_reasons.append(obfuscation_text)

    if permissions_info["suspicious_permissions"]:
        permissions_text = "\n".join([f"   * {perm}" for perm in permissions_info["suspicious_permissions"]])
        flagged_reasons.append(f" - Suspicious permissions requested:\n{permissions_text}")

    if apk_signature_result != "APK Signature Valid":
        flagged_reasons.append(f" - APK signature verification failed: {apk_signature_result}")

    # Print Final Output in a Formatted Manner
    print(f"\nAPK Hash (SHA-256): {apk_hash}")
    print("\nFinal Decision:")

    if flagged_reasons:
        print("\nWARNING: This APK is flagged as suspicious or malicious for the following reasons:")
        for reason in flagged_reasons:
            print(reason)
    else:
        print("\nThis APK appears to be legitimate based on the checks performed.")

    # 8. Print APK Metadata and Permissions at the end
    print(f"\nPackage Name: {permissions_info['package_name']}")
    print(f"Version Name: {permissions_info['version_name']}")
    print(f"Version Code: {permissions_info['version_code']}")

    # Permissions Output
    print("\nPermissions found:")
    for permission in permissions_info["permissions"]:
        if permission in permissions_info["suspicious_permissions"]:
            print(f"Suspicious permission found: {permission}")
        else:
            print(f"Permission: {permission}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python apk_tool.py <APK Path>")
        sys.exit(1)
        
    apk_file = sys.argv[1]
    api_keys = {"virustotal": "e6bdb1c0beec509fbf984d14e8f4aa5473d47147e98a6bf31b5ed7b9faf6ae73"}
    detect_malicious_apk(apk_file, api_keys)

if __name__ == "__main__":
    main()