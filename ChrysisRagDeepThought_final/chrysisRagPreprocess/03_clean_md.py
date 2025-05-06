import os
import re
from pathlib import Path

# Define input and output directories
md_dir = Path("02_md_files")
cleaned_md_dir = Path("03_cleaned_md_files")

# Create the output directory if it doesn't exist
cleaned_md_dir.mkdir(parents=True, exist_ok=True)

print(f"Cleaning Markdown files from '{md_dir}' to '{cleaned_md_dir}'...")

# NOT USED CURRENYTLY BUT PRESET FOR FUTURE USE
# Regex patterns to identify and remove common elements
# Pattern to find the main navigation block (often starting with HOME or ΑΡΧΙΚΗ)
nav_link_pattern = re.compile(r"^\*\s*\[[^\]]+\]\([^\)]+\)(?:\s+\"[^\"]*\"?)?$")
# Pattern for image links often used as logos/banners
image_link_pattern = re.compile(r"^\[!\[[^\]]*\]\([^\)]+\)\]\([^\)]+\)$")
# Pattern for simple list item links (often part of nav)
simple_list_link_pattern = re.compile(r"^\*\s+\[[^\]]+\]\([^\)]+\)$")
# Pattern for multi-level list item links
multi_level_list_link_pattern = re.compile(r"^\s*\+\s+\[[^\]]+\]\([^\)]+\)$")
# Pattern for language links like * [EN](...) or * [EL](...)
lang_link_pattern = re.compile(r"^\*\s*\[(?:EL|EN)\]\([^\)]+\)(?:\s+\"[^\"]*\"?)?$")
# Pattern for jump-to-content link
jump_link_pattern = re.compile(r"^\[Μετάβαση στο περιεχόμενο\]\(#content\)$")
# NOT USED CURRENYTLY BUT PRESET FOR FUTURE USE

# Footer patterns
contact_header_pattern = re.compile(r"^#### Contact Us$")
subscribe_header_pattern = re.compile(r"^#### Subscribe now !$")
calendar_header_pattern = re.compile(r"^#### Calendar$")
latest_news_header_pattern = re.compile(r"^#### Latest News$")
social_media_header_pattern = re.compile(r"^#### Social Media$")
copyright_pattern = re.compile(r"^©\s+(?:University of Cyprus|UCY Library)")
cookie_banner_pattern = re.compile(r"^(?:##### \*\*)?University of Cyprus(?:\*\*)? Website Cookies")
privacy_overview_pattern = re.compile(r"^#### Privacy Overview$")
accessibility_tools_pattern = re.compile(r"^Εργαλεία προσβασιμότητας$")
toggle_sliding_bar_pattern = re.compile(r"^\[Toggle Sliding Bar Area\]\(#\)$")
h1_contact_us_pattern = re.compile(r"^# Contact Us$")

# Function to clean a single markdown file's content
def clean_markdown(content):
    lines = content.split('\n')
    
    # Find the index of the first H1 header ('# ')
    header_end_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('# '): # Be specific to H1
            header_end_index = i
            break
            
    # If no H1 header found, maybe the file is structured differently or empty
    # For aggressive cleaning, we'll return empty content in this case.
    if header_end_index == -1:
        # print(f"Warning: No H1 ('# ') header found. Returning empty content.") 
        return ""

    # Consider lines from the first H1 header onwards
    content_lines = lines[header_end_index:]

    # Find the start of the footer within the content lines
    potential_footer_start = len(content_lines)
    for i, line in enumerate(content_lines):
        stripped_line = line.strip()
        if any(p.match(stripped_line) for p in [
            h1_contact_us_pattern,
            contact_header_pattern, subscribe_header_pattern, calendar_header_pattern,
            latest_news_header_pattern, social_media_header_pattern, copyright_pattern,
            cookie_banner_pattern, privacy_overview_pattern, accessibility_tools_pattern,
            toggle_sliding_bar_pattern
        ]):
            potential_footer_start = i
            break # Stop at the first sign of a footer pattern

    # Keep lines from the H1 header up to (but not including) the footer start
    cleaned_lines = content_lines[:potential_footer_start]

    # Strip leading/trailing whitespace and excessive newlines
    final_content = '\n'.join(cleaned_lines).strip()
    # Reduce multiple consecutive blank lines to a single blank line
    final_content = re.sub(r'\n{3,}', '\n\n', final_content)

    return final_content


# Iterate through files in the md directory
for md_file_path in md_dir.glob("*.md"):
    if md_file_path.is_file():
        try:
            # Read Markdown content
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # Clean Markdown content
            cleaned_content = clean_markdown(md_content)

            # Determine the output markdown file path
            cleaned_md_file_path = cleaned_md_dir / md_file_path.name

            # Write cleaned Markdown content to the output file
            with open(cleaned_md_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            print(f"Cleaned '{md_file_path.name}' to '{cleaned_md_file_path.name}'")

        except Exception as e:
            print(f"Error cleaning file '{md_file_path.name}': {e}")

print("Cleaning process finished.") 