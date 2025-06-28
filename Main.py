!pip install transformers gradio

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import gradio as gr

# Loads the model
model_name = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Sample context
context = """
1. Asthma -Asthma is a chronic respiratory condition where the airways become inflamed and narrow, causing wheezing, coughing, chest tightness, and shortness of breath.
Triggers include allergens, exercise, and cold air.
2. Anemia-Anemia is a condition in which there are not enough healthy red blood cells to carry oxygen to the body’s tissues.
Common symptoms include fatigue, weakness, pale skin, and shortness of breath.
3. Arthritis-Arthritis refers to inflammation of one or more joints, causing pain, swelling, and stiffness.
 Osteoarthritis and rheumatoid arthritis are the most common types.
4. Tuberculosis (TB)-Tuberculosis is an infectious disease caused by Mycobacterium tuberculosis, primarily affecting the lungs.
 Symptoms include persistent cough, night sweats, weight loss, and fever.
5. Migraine-A migraine is a neurological disorder that causes intense, throbbing headaches, often accompanied by nausea, vomiting, and sensitivity to light or sound.
6. Cancer-Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells.
 It can occur in any part of the body and may require surgery, radiation, or chemotherapy.
7. Obesity-Obesity is a complex condition involving excessive body fat that increases the risk of heart disease, diabetes, and certain cancers.
It is often managed through diet, exercise, and behavioral changes.
8. Stroke-A stroke occurs when blood flow to the brain is interrupted, leading to brain damage.
 Symptoms include sudden numbness, confusion, trouble speaking, and loss of coordination.
9. Depression-Depression is a mood disorder marked by persistent sadness, lack of interest, fatigue, and changes in appetite or sleep.
 Treatment may involve therapy, medication, or both.
10. Anxiety Disorders-These include conditions like generalized anxiety, panic disorder, and phobias.
 Symptoms may include restlessness, excessive worry, rapid heartbeat, and trouble concentrating.
"""

# Functionality
def answer_question(question):
    if not question.strip():
        return "Please enter a medical question."
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    return answer if answer.strip() else "I'm sorry, I couldn't find an answer."

# Gradio Interface
gr.Interface(fn=answer_question,
             inputs=gr.Textbox(lines=2, placeholder="Ask a health question..."),
             outputs="text",
             title="Healthcare AI Assistant",
             description="Ask health-related questions. Informational use only.").launch()
