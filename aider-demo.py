import PyPDF2

def read_pdf_text(filename):
    with open(filename, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text

def main():
    filename = "demo.pdf"
    text = read_pdf_text(filename)
    print(text)

if __name__ == '__main__':
    main()
