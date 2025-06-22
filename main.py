import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return {label: score}


gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs = gr.Label(num_top_classes=1),
    title="Sentiment Analyzer"
).launch()
