from transformers import pipeline
from flask import Flask, render_template, request

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Initialize pipelines

ner = pipeline('ner', aggregation_strategy="average")  # Use aggregation_strategy for better output

@app.route("/", methods=["GET", "POST"])
def index():
    
    entities = []
    
    if request.method == "POST":
        text = request.form["text"]


        # NER Processing
        entities = ner(text)

    return render_template("index.html", entities=entities)

if __name__ == "__main__":
    app.run(debug=True)
