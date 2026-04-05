import gradio as gr
from models import get_triage_model
from typing import Dict

# Initialize the singleton model
triage_model = get_triage_model()

def predict(email_text: str) -> Dict[str, str]:
    """
    Main triage function for deployment.
    Categorizes the email into category, priority, and department.
    """
    if not email_text or len(email_text.strip()) == 0:
        return {"error": "Input text is empty"}
        
    try:
        prediction = triage_model.predict(email_text)
        return prediction
    except Exception as e:
        return {"error": str(e)}

def triage_interface(text):
    """Gradio wrapper for predict"""
    result = predict(text)
    return result

# Gradio UI Setup
def launch_app():
    with gr.Blocks(title="AI Email Triage System") as demo:
        gr.Markdown("# 📧 AI Email Triage System")
        gr.Markdown("Automated classification using DistilBART Zero-Shot classification.")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Email Content", 
                    placeholder="Enter email text here...", 
                    lines=5
                )
                submit_btn = gr.Button("Analyze Email", variant="primary")
            
            with gr.Column():
                output_json = gr.JSON(label="Analysis Results")
        
        gr.Examples(
            examples=[
                ["URGENT: Our production server is down! We need immediate help."],
                ["Could you please check my vacation policy for next month?"],
                ["Hey, just wanted to follow up on the sales lead from yesterday."],
                ["Inquiry regarding our last billing statement and unpaid invoice."]
            ],
            inputs=input_text
        )
        
        submit_btn.click(fn=triage_interface, inputs=input_text, outputs=output_json)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    launch_app()
