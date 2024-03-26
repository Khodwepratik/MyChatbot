from flask import Flask, render_template, request, jsonify
import json
import fitz
import spacy
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM

app = Flask(__name__)

# Constants declaration
PDF_PATH = 'java.pdf'
CHUNK_SIZE = 500
nlp = spacy.load("en_core_web_sm")
ner_model_name = "dslim/bert-base-NER"
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
text_generation_model_name = "openai-community/gpt2"
text_generation_model = AutoModelForCausalLM.from_pretrained(text_generation_model_name)
text_generation_tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)
classification_model_name = "google-bert/bert-base-uncased"
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)

# Custom Exceptions
class PDFExtractionError(Exception):
    pass

class BotResponseError(Exception):
    pass

# Load data from a JSON file
def load_json_data(file_path='data.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("JSON file not found. Please make sure the file exists.")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        raise PDFExtractionError("Error extracting text from PDF. Please check the PDF file.") from e

# Named Entity Recognition (NER) with entity linking and disambiguation
def recognize_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# Function to generate text using fine-tuned GPT-3 model
def generate_text(query):
    inputs = text_generation_tokenizer("generate text: " + query, return_tensors="pt", max_length=512, truncation=True)
    outputs = text_generation_model.generate(inputs.input_ids, max_length=300, num_return_sequences=1, temperature=1.7)
    return text_generation_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Text classification
def classify_text(text):
    inputs = classification_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = classification_model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class

def get_pdf_answer_bert(query, pdf_text, max_length=200):
    relevant_text = ""
    chunks = [pdf_text[i:i + max_length] for i in range(0, len(pdf_text), max_length)]
    for chunk in chunks:
        if query.lower() in chunk.lower():
            relevant_text += chunk

    if relevant_text:
        try:
            summarizer = pipeline("summarization")
            summary = summarizer(relevant_text, max_length=500, min_length=50, length_penalty=2.0, num_beams=2)
            return summary[0]['summary_text']
        except Exception as e:
            raise BotResponseError("Error generating summary using BERT pipeline.") from e
    else:
        return None
    
# Bot response generation
def generate_bot_response(user_input, entities, json_data):
    try:
        best_answer = None
        max_score = 0
        for item in json_data['questions']:
            score = sum(1 for entity, _ in entities if entity.lower() in item['question'].lower())
            if score > max_score:
                best_answer = item['answer']
                max_score = score

        if best_answer:
            return best_answer
        
    except (PDFExtractionError, BotResponseError) as e:
        return str(e)
    except Exception as e:
        return "An unexpected error occurred. Please try again later."

    try:
        pdf_text = extract_text_from_pdf(PDF_PATH)

        # Handling questions like "What is_____?"
        if user_input.lower().startswith("what is ") and user_input.endswith("?"):
            entity = user_input[8:-1].strip()
            if entity.lower() in pdf_text.lower():
                return get_pdf_answer_bert(entity, pdf_text)
            else:
                return "Sorry, I don't have information about " + entity

        # Handling questions like "Who is ____?"
        if user_input.lower().startswith("who is ") and user_input.endswith("?"):
            entity = user_input[7:-1].strip()
            if entity.lower() in pdf_text.lower():
                return get_pdf_answer_bert(entity, pdf_text)
            else:
                return "Sorry, I don't have information about " + entity

        # Handling enlisting type of question
        if user_input.lower().startswith("enlist "):
            list_query = user_input.split(" ", 1)[1]

            if list_query.lower() in pdf_text.lower():
                items = get_pdf_answer_bert("Enlist " + list_query, pdf_text)
                if items:
                    formatted_items = "\n".join([f"- {item}" for item in items.split("\n")])
                    return f"Here is a list of {list_query}:\n{formatted_items}"
                else:
                    return f"Sorry, I didn't found information about {list_query}"
            else:
                return f"Sorry, I don't have information about {list_query}"

        # Handling "How to" questions
        if user_input.lower().startswith("how to ") and user_input.endswith("?"):
            entity = user_input[7:-1].strip()
            if entity.lower() in pdf_text.lower():
                return get_pdf_answer_bert(entity, pdf_text)
            else:
                return "Sorry, I don't have information about how to " + entity

        # Handling questions like "Define _____?"
        if user_input.lower().startswith("define "):
            word = user_input[7:-1].strip()
            if word.lower() in pdf_text.lower():
                return get_pdf_answer_bert(word, pdf_text)
            else:
                return f"Sorry, I don't have information about {word}"

        pdf_answer = get_pdf_answer_bert(user_input, pdf_text)
        if pdf_answer:
            return pdf_answer

        # If no direct answer found, use text generation
        generated_text = generate_text(user_input)
        return generated_text

    except (PDFExtractionError, BotResponseError) as e:
        return str(e)
    except Exception as e:
        return "An unexpected error occurred. Please try again later."

# Routes for flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        user_input = request.form['user_input']
        entities = recognize_entities(user_input)
        response = generate_bot_response(user_input, entities, load_json_data())
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
