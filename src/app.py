import os
from flask import Flask, request, render_template, jsonify
from mcqgenerator.mcqgenerator import generate_mcqs  # Adjusted to use the modified function
from werkzeug.utils import secure_filename
import chardet

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility function to split text into chunks
def split_text_into_chunks(text, max_chunk_size=2000):
    # Split the text into smaller chunks to fit the token limit
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # Account for space between words
        if current_size >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add remaining words as a chunk
    return chunks

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        file = request.files["file"]
        number = request.form.get("number")
        subject = request.form.get("subject")
        tone = request.form.get("tone")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Read file and detect encoding
        with open(file_path, "rb") as raw_file:
            raw_data = raw_file.read()
            result_encoding = chardet.detect(raw_data)
            encoding = result_encoding["encoding"] if result_encoding["encoding"] else "utf-8"
            try:
                text = raw_data.decode(encoding)
            except UnicodeDecodeError:
                text = raw_data.decode("ISO-8859-1")  # Fallback if the default fails

        # Split the large text into chunks
        chunks = split_text_into_chunks(text)

        # Generate MCQs for each chunk of text
        mcq_results = []
        for chunk in chunks:
            result = generate_mcqs(chunk, number, subject, tone)
            mcq_results.append(result)

        # Combine results from all chunks
        combined_results = {
            "quiz": [result['quiz'] for result in mcq_results],
            "review": [result['review'] for result in mcq_results],
        }

        return jsonify(combined_results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
