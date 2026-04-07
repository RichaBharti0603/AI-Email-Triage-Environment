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
        return {"category": "Normal", "priority": "Low", "department": "HR"}
        
    try:
        prediction = triage_model.predict(email_text)
        return prediction.dict() if hasattr(prediction, 'dict') else prediction
    except Exception as e:
        return {"error": str(e)}

def triage_display(text):
    """Gradio wrapper for predict with formatted HTML output"""
    result = predict(text)
    if "error" in result:
        return f"### Error: {result['error']}"
    
    # Define color mappings for aesthetic badges (matching OpenEnv capitalization)
    colors = {
        'Urgent': '#f44336', 'Normal': '#2196f3', 'Spam': '#9e9e9e',
        'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50',
        'Tech': '#673ab7', 'HR': '#e91e63', 'Sales': '#ffeb3b', 'Billing': '#00bcd4'
    }
    
    cat = str(result.get('category', 'Normal'))
    pri = str(result.get('priority', 'Low'))
    dep = str(result.get('department', 'HR'))
    
    # Premium Triage Card using HTML
    html_card = f"""
    <div style="background: #1a1a1a; padding: 20px; border-radius: 12px; border-left: 5px solid {colors.get(pri, '#fff')}; color: white; font-family: 'Inter', sans-serif;">
        <h3 style="margin-top: 0; color: #eee;">📧 Email Analysis Result</h3>
        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px;">
            <span style="background: {colors.get(cat, '#333')}; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">Category: {cat.upper()}</span>
            <span style="background: {colors.get(pri, '#333')}; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">Priority: {pri}</span>
            <span style="background: {colors.get(dep, '#333')}; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">Dept: {dep}</span>
        </div>
        <div style="font-size: 0.9em; opacity: 0.8; line-height: 1.5;">
            {text[:200]}{'...' if len(text) > 200 else ''}
        </div>
    </div>
    """
    return html_card

# Gradio UI Setup
def launch_app():
    with gr.Blocks(title="AI Email Triage System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📧 AI Email Triage System")
        gr.Markdown("Automated classification using DistilBART-MNLI (Optimized for CPU).")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Email Content", 
                    placeholder="Enter email text here...", 
                    lines=8
                )
                submit_btn = gr.Button("🚀 Analyze Email", variant="primary")
            
            with gr.Column(scale=1):
                output_html = gr.HTML(label="Triage Card")
        
        gr.Examples(
            examples=[
                ["CRITICAL: Production database down. Users reporting 502 errors."],
                ["Inquiry about my last invoice #4422. It looks like double billing."],
                ["Congratulations! You won a free iPhone 15. Click here to claim your prize!"],
                ["Hey, I would like to schedule a vacation from June 10-15. How do I apply?"]
            ],
            inputs=input_text
        )
        
        submit_btn.click(fn=triage_display, inputs=input_text, outputs=output_html)

    # Launch app - share=False for local deployment
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    launch_app()
