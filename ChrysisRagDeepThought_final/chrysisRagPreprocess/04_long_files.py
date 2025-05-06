import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def count_words(file_path):
    """Counts the words in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            words = content.split()
            return len(words)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return 0 # Treat files that cannot be read as having 0 words

def copy_long_files(source_directory, destination_directory, min_word_count=100):
    """
    Copies files from the source directory to the destination directory
    if their word count is greater than or equal to min_word_count.
    The source directory remains unchanged.
    """
    if not os.path.isdir(source_directory):
        logging.error(f"Source directory not found: {source_directory}")
        return

    # Create destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)
    logging.info(f"Ensured destination directory exists: {destination_directory}")

    copied_count = 0
    processed_count = 0

    logging.info(f"Starting file processing in directory: {source_directory}")
    for filename in os.listdir(source_directory):
        source_file_path = os.path.join(source_directory, filename)
        if os.path.isfile(source_file_path):
            processed_count += 1
            word_count = count_words(source_file_path)
            logging.debug(f"File: {filename}, Word Count: {word_count}")

            # Check if the word count is greater than or equal to the minimum
            if word_count >= min_word_count:
                destination_file_path = os.path.join(destination_directory, filename)
                try:
                    # Use copy2 to preserve metadata like modification time
                    shutil.copy2(source_file_path, destination_file_path)
                    logging.info(f"Copied long file: {filename} to {destination_directory} (Word count: {word_count})")
                    copied_count += 1
                except Exception as e:
                    logging.error(f"Error copying file {source_file_path} to {destination_file_path}: {e}")
            else:
                 logging.debug(f"Skipping short file: {filename} (Word count: {word_count})")

    logging.info(f"Finished processing.")
    logging.info(f"Total files processed: {processed_count}")
    logging.info(f"Total files copied: {copied_count}")

if __name__ == "__main__":
    source_dir = "03_cleaned_md_files"
    # Define the directory to copy long files to
    destination_dir = os.path.join(os.path.dirname(source_dir), "long_files")

    copy_long_files(source_dir, destination_dir) 