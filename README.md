#  Health Care Assistant

An AI-powered healthcare chatbot built using Hugging Face Transformers and Gradio. This assistant helps users ask medical questions and get context-aware answers based on biomedical literature.



##  Project Objective

To create a simple and interactive chatbot that assists patients or users by answering health-related queries based on real medical data, using a pre-trained transformer model fine-tuned on biomedical datasets.



##  Features

-  Natural Language Question Answering
-  Powered by BioBERT (`ktrapeznikov/biobert_v1.1_pubmed_squad_v2`)
-  Context-aware medical responses
-  User-friendly web interface with **Gradio**
-  Real-time answers based on provided medical context
-  Runs entirely on local/cloud server — no data is stored



##  Tech Stack

| Component       | Tech Used                             |
|-----------------|---------------------------------------|
| Model           | Hugging Face Transformers, BioBERT    |
| Interface       | Gradio                                |
| Programming Lang| Python                                |



##  How It Works

1. User enters a health-related question
2. The model is queried using a predefined medical context
3. BioBERT extracts the most relevant answer from the context
4. The answer is shown on the interface via Gradio


## Sample Context Used

```text
Asthma is a chronic respiratory condition that causes difficulty in breathing due to inflammation and narrowing of the airways. Symptoms include wheezing, coughing, chest tightness, and shortness of breath. Treatment involves inhalers, corticosteroids, and avoiding triggers.
