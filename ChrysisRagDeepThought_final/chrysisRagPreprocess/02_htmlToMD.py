import os
import markdownify
from pathlib import Path

# Define input and output directories
html_dir = Path("01_html_files")
md_dir = Path("02_md_files")

# Create the output directory if it doesn't exist
md_dir.mkdir(parents=True, exist_ok=True)

print(f"Converting HTML files from '{html_dir}' to Markdown in '{md_dir}'...")

# Iterate through files in the html directory
for html_file_path in html_dir.glob("*.html"):
    if html_file_path.is_file():
        try:
            # Read HTML content
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Convert HTML to Markdown
            md_content = markdownify.markdownify(html_content, heading_style="ATX")

            # Determine the output markdown file path
            md_file_name = html_file_path.stem + ".md"
            md_file_path = md_dir / md_file_name

            # Write Markdown content to the output file
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            print(f"Converted '{html_file_path.name}' to '{md_file_path.name}'")

        except Exception as e:
            print(f"Error converting file '{html_file_path.name}': {e}")

# Also handle .htm files if they exist
for htm_file_path in html_dir.glob("*.htm"):
     if htm_file_path.is_file():
        try:
            # Read HTML content
            with open(htm_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Convert HTML to Markdown
            md_content = markdownify.markdownify(html_content, heading_style="ATX")

            # Determine the output markdown file path
            md_file_name = htm_file_path.stem + ".md"
            md_file_path = md_dir / md_file_name

            # Write Markdown content to the output file
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            print(f"Converted '{htm_file_path.name}' to '{md_file_path.name}'")

        except Exception as e:
            print(f"Error converting file '{htm_file_path.name}': {e}")


print("Conversion process finished.")
