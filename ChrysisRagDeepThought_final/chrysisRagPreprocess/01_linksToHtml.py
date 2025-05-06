import csv
import requests
import os
import re
from urllib.parse import urlparse
import warnings

# Suppress only the InsecureRequestWarning from urllib3 needed for verify=False
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# Function to sanitize URL to create a valid filename
def sanitize_filename(url):
    """
    Converts a URL into a safe filename.
    Removes scheme, replaces invalid characters, limits length.
    """
    try:
        parsed_url = urlparse(url)
        # Handle cases where URL might be just a domain or malformed
        netloc = parsed_url.netloc if parsed_url.netloc else url
        path = parsed_url.path if parsed_url.path else ''
        filename = netloc + path
    except ValueError:
        # Fallback for completely unparseable URLs
        filename = url

    # Replace invalid filesystem characters with underscores
    filename = re.sub(r'[<>:"/\|?*]', '_', filename)
    # Replace multiple consecutive underscores with a single one
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores or slashes if any resulted
    filename = filename.strip('_/')
    # Limit length to avoid issues with long filenames
    max_len = 150
    if len(filename) > max_len:
        # Try to keep the end part, might be more informative
        filename = filename[-max_len:]
        # Ensure it doesn't start with an underscore after truncation
        filename = filename.lstrip('_')

    # If filename becomes empty after sanitization, use a placeholder
    if not filename:
        filename = 'default_filename'

    return filename + ".html"

# --- Configuration ---
CSV_FILE_PATH = 'links.csv'
OUTPUT_DIR = 'html_files'
REQUEST_TIMEOUT = 45  # seconds

# --- Load Bright Data Credentials ---
brightdata_host = os.environ.get("BRIGHTDATA_HOST")
brightdata_port = os.environ.get("BRIGHTDATA_PORT")
brightdata_username = os.environ.get("BRIGHTDATA_USERNAME")
brightdata_password = os.environ.get("BRIGHTDATA_PASSWORD")

if not all([brightdata_host, brightdata_port, brightdata_username, brightdata_password]):
    print("Error: Bright Data credentials not fully configured.")
    print("Please set BRIGHTDATA_HOST, BRIGHTDATA_PORT, BRIGHTDATA_USERNAME, and BRIGHTDATA_PASSWORD environment variables.")
    exit(1)

# --- Construct Proxy URL ---
proxy_url = f"http://{brightdata_username}:{brightdata_password}@{brightdata_host}:{brightdata_port}"
proxies = {
    "http": proxy_url,
    "https": proxy_url,
}

# --- Main Execution ---
def main():
    """
    Reads URLs from CSV, fetches HTML via proxy, saves to files.
    """
    # Create output directory if it doesn't exist
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{OUTPUT_DIR}': {e}")
        exit(1)

    print(f"Reading URLs from: {CSV_FILE_PATH}")
    print(f"Saving HTML files to: {OUTPUT_DIR}")
    print(f"Using Bright Data proxy: {brightdata_host}:{brightdata_port}")

    urls_to_process = []
    try:
        with open(CSV_FILE_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row: # Ensure row is not empty
                    url = row[0].strip()
                    if url: # Ensure URL string is not empty
                        urls_to_process.append(url)
    except FileNotFoundError:
        print(f"Error: Input file not found at {CSV_FILE_PATH}")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{CSV_FILE_PATH}': {e}")
        exit(1)

    if not urls_to_process:
        print("No valid URLs found in the CSV file.")
        exit(0)

    print(f"Found {len(urls_to_process)} URLs to process.")

    processed_count = 0
    error_count = 0
    for i, url in enumerate(urls_to_process):
        if not url.startswith(('http://', 'https://')):
            print(f"Skipping invalid URL format ({i+1}/{len(urls_to_process)}): {url}")
            error_count += 1
            continue

        print(f"Fetching ({i+1}/{len(urls_to_process)}): {url}")
        try:
            # Using verify=False to ignore SSL errors, common with proxies. 
            response = requests.get(url, proxies=proxies, timeout=REQUEST_TIMEOUT, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Sanitize URL to create a filename
            filename = sanitize_filename(url)
            output_path = os.path.join(OUTPUT_DIR, filename)

            # Save HTML content
            try:
                with open(output_path, 'w', encoding='utf-8') as htmlfile:
                    htmlfile.write(response.text)
                print(f"  -> Saved to {output_path}")
                processed_count += 1
            except IOError as e:
                print(f"  -> Error writing file {output_path}: {e}")
                error_count += 1
            except Exception as e: # Catch other potential errors during file write
                print(f"  -> Unexpected error writing file {output_path}: {e}")
                error_count += 1


        except requests.exceptions.Timeout:
            print(f"  -> Error: Request timed out for {url}")
            error_count += 1
        except requests.exceptions.ConnectionError:
            print(f"  -> Error: Could not connect to {url}")
            error_count += 1
        except requests.exceptions.HTTPError as e:
            print(f"  -> Error: HTTP Error {e.response.status_code} for {url}")
            error_count += 1
        except requests.exceptions.RequestException as e:
            print(f"  -> Error: Failed to fetch {url}: {e}")
            error_count += 1
        except Exception as e:
            print(f"  -> An unexpected error occurred for {url}: {e}")
            error_count += 1

    print("\n--- Processing Summary ---")
    print(f"Total URLs attempted: {len(urls_to_process)}")
    print(f"Successfully saved:   {processed_count}")
    print(f"Errors encountered:   {error_count}")
    print("--------------------------")

if __name__ == "__main__":
    main()
