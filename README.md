# MyChatbot
Questions and answering chatbot with pdf in python 

## Dependencies
1. Flask: A micro web framework for Python.
2. fitz: A Python wrapper for PyMuPDF, which enables PDF parsing.
3. spacy: An open-source NLP library.
4. transformers: A library by Hugging Face for natural language understanding tasks.


## Installation
1. Ensure Python is installed on your system.
2. Install required Python packages using pip:
 
  pip install flask pymupdf spacy transformers
  python -m spacy download en_core_web_sm

3. Place the PDF file (java.pdf by default) in the project directory.
4. Optionally, modify the default JSON data file (data.json) with custom questions and answers.


## Execute

Run the file using 
    python app.py 

1. Access the chatbot interface through a web browser at http://127.0.0.1:5000/.
2. Enter your queries in the input field provided.
3. Press Enter or click the Submit button to receive responses from the chatbot.
4. The chatbot utilizes NLP techniques to understand the query and responds with relevant information.
5. It can also extract information from the provided PDF file based on the query.

## Features
1. PDF Text Extraction: The application extracts text from a PDF file (java.pdf by default) using the fitz library.
2. User Input Analysis: User input is analyzed using the spacy library to identify entities and tokens in the query.
3. Bot Response Generation: Based on the user query, the bot provides responses by matching against pre-defined questions in a JSON file (data.json by default) or by extracting information from the PDF file.
4. Bert-based Answer Extraction: For PDF-related queries, the application employs BERT-based summarization using the transformers library to generate concise answers.
