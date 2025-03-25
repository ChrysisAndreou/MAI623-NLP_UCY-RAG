'''I created this code to extract information from scheduled PDFs
and automatically generate questions and answers. Every time, you
choose the PDF that it will read.
It then generates two CSV files: the first contains the extracted
data from the PDF using ReGex, and the second contains the generated
questions and answers.

'''
from pathlib import Path
import pdfplumber
import csv
import re

# File paths
PDF_FILE_PATH = Path("epa.pdf")
CSV_FILE_PATH = Path("output.csv")
QA_CSV_FILE_PATH = Path("questions_and_answers.csv")

# Initialize data storage
columns = {
    'CRN': [],
    'Course Department': [],
    'Course Number': [],
    'Section': [],
    'Extra Section': [],
    'Course Title': [],
    'ECTS': [],
    'Days': [],
    'Time': [],
    'Building': [],
    'Room': [],
    'Capacity': [],
    'Seats': [],
    'Instructor': []
}

# Regex pattern to match course lines
course_line_pattern = re.compile(r'''
    (?P<crn>\d{4,6})\s+
    (?P<dept>[A-Z]{2,4})\s+
    (?P<number>\d{3})\s+
    (?P<section>[A-Z0-9Α-Ω]+)?\s*
    (?P<extra_section>[Α-Ω]{1,3})?\s*
    (?P<title>.+?)\s+
    (?P<ects>\d+(?:\.\d+)?)\s+
    (?P<days>[MTWRFS\.\s]{5,20})\s+
    (?P<time>\d{4}-\d{4})?\s*
    (?P<building>[A-ZΑ-Ω]{2,5}\d{0,3})?\s*
    (?P<room>[A-ZΑ-Ω]?\d{2,5})?\s*
    (?P<cap>\b\d{1,3}\b)?\s*
    (?P<seats>\d{1,3})?\s*
    (?P<instructor>[A-ZΑ-Ω][a-zα-ω]+(?:\s+[A-ZΑ-Ω][a-zα-ω]+)+)?
''', re.VERBOSE)
# Convert day dot pattern to readable day list
DAY_MAP = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

def interpret_days(dot_pattern: str) -> str:
    chars = dot_pattern.strip().replace(' ', '')
    if len(chars) != 6:
        return dot_pattern  # fallback if pattern unexpected
    return ", ".join([DAY_MAP[i] for i, c in enumerate(chars) if c != '.'])

# Extract and parse PDF content
with pdfplumber.open(PDF_FILE_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            match = course_line_pattern.match(line)
            if match:
                columns['CRN'].append(match.group('crn') or '')
                columns['Course Department'].append(match.group('dept') or '')
                columns['Course Number'].append(match.group('number') or '')
                columns['Section'].append(match.group('section') or '')
                columns['Extra Section'].append(match.group('extra_section') or '')
                columns['Course Title'].append(match.group('title') or '')
                columns['ECTS'].append(match.group('ects') or '')
                raw_days = (match.group('days') or '').strip()
                readable_days = interpret_days(raw_days)
                columns['Days'].append(readable_days)
                columns['Time'].append(match.group('time') or '')
                columns['Building'].append(match.group('building') or '')
                columns['Room'].append(match.group('room') or '')
                columns['Capacity'].append(match.group('cap') or '')
                columns['Seats'].append(match.group('seats') or '')
                columns['Instructor'].append((match.group('instructor') or '').strip())

# Write data to CSV
with CSV_FILE_PATH.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(columns.keys())
    writer.writerows(zip(*columns.values()))

print(f"CSV file '{CSV_FILE_PATH}' created with {len(columns['CRN'])} vertical rows.")

# Generate multilingual Q&A CSV (3 columns: original line, question, answer)
qa_rows = []

for i in range(len(columns['CRN'])):
    crn = columns['CRN'][i]
    title = columns['Course Title'][i]
    instructor = columns['Instructor'][i] or "Δεν έχει καθοριστεί"
    days = columns['Days'][i]
    time = columns['Time'][i] or "Δεν έχει καθοριστεί ώρα"
    building = columns['Building'][i]
    room = columns['Room'][i]
    ects = columns['ECTS'][i]

    original_line = f"{crn} {columns['Course Department'][i]} {columns['Course Number'][i]} {columns['Section'][i]} {columns['Extra Section'][i]} {title} {ects} {days} {time} {building} {room} {columns['Capacity'][i]} {columns['Seats'][i]} {instructor}".strip()

    qas = [
        # English
        (f"What is the title of the course with CRN {crn}?", f"The title of the course with CRN {crn} is '{title}.'"),
        (f"Who teaches {title}?", f"This course, titled '{title}', is taught by {instructor}."),
        (f"When is {title} scheduled?", f"This course, titled '{title}', is scheduled on {days} at {time}."),
        (f"Where is {title} held?", f"This course, titled '{title}', is held in {building} room {room}."),
        (f"What are the ECTS credits for {title}?", f"This course, titled '{title}', is worth {ects} ECTS credits."),
        # Greek - variant phrasings
        (f"Ποιος είναι ο τίτλος του μαθήματος με CRN {crn};", f"Ο τίτλος του μαθήματος με CRN {crn} είναι '{title}.'"),
        (f"Ποιος διδάσκει το μάθημα {title};", f"Το μάθημα '{title}' διδάσκεται από τον/την {instructor}."),
        (f"Ποιος είναι ο διδάσκων του μαθήματος '{title}';", f"Ο διδάσκων του μαθήματος '{title}' είναι ο/η {instructor}."),
        (f"Πότε πραγματοποιείται το μάθημα '{title}';", f"Το μάθημα '{title}' πραγματοποιείται τις ημέρες {days}, στις {time}."),
        (f"Πού γίνεται το μάθημα '{title}';", f"Το μάθημα '{title}' γίνεται στο κτίριο {building}, αίθουσα {room}."),
        (f"Πόσες μονάδες ECTS έχει το μάθημα '{title}';", f"Το μάθημα '{title}' αντιστοιχεί σε {ects} μονάδες ECTS."),
    ]

    for q, a in qas:
        qa_rows.append([PDF_FILE_PATH.name, q, a])

with QA_CSV_FILE_PATH.open('w', newline='', encoding='utf-8-sig') as qa_file:
    writer = csv.writer(qa_file)
    writer.writerow(['source_pdf', 'question', 'answer'])
    writer.writerows(qa_rows)

print(f"Q&A CSV file '{QA_CSV_FILE_PATH}' created with {len(qa_rows)} rows.")
