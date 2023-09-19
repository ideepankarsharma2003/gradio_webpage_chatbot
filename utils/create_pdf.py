from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet



def create_pdf(paragraphs):
    output_pdf = "output.pdf"
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)

    # Create a list of paragraphs for the content
    story = []

    # Set font and font size
    styles = getSampleStyleSheet()
    style = styles['Normal']

    # Add each string as a paragraph to the PDF
    for string in paragraphs:
        p = Paragraph(string, style)
        story.append(p)

    # Build the PDF document
    doc.build(story)

    print(f"PDF file '{output_pdf}' has been created.")