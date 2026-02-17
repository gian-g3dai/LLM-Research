import io

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# Define the input PDF file and the output Markdown file
input_pdf = 'dscsharp.pdf'
output_md = 'coding_textbook.md'

# Function to extract text from a range of pages
def extract_text_from_pages(pdf_file, start_page, end_page):
    output = io.StringIO()
    laparams = LAParams()
    with open(pdf_file, 'rb') as fp:
        extract_text(fp, output, laparams=laparams, page_numbers=range(start_page, end_page + 1))
    return output.getvalue()

# Extract text from the first 10 pages
text = extract_text_from_pages(input_pdf, start_page=1, end_page=50)

# Split text into paragraphs
paragraphs = text.split('\n\n')

# Open the output Markdown file for writing
with open(output_md, 'w') as md_file:
    for paragraph in paragraphs:
        # Check if the paragraph looks like code (you might need to adjust this condition)
        is_code = any(line.strip().startswith(('>>> ', 'In [', 'Out [')) for line in paragraph.split('\n'))

        # Write the paragraph to the Markdown file with appropriate formatting
        if is_code:
            md_file.write('```\n')
            md_file.write(paragraph)
            md_file.write('\n```\n')
        else:
            md_file.write(paragraph + '\n')
