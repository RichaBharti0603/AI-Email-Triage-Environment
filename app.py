import gradio as gr
from models import get_triage_model
from environment import EmailTriageEnv
from grader import evaluate
from typing import Dict, Any

# Initialize the singleton model and env
triage_model = get_triage_model()
env = EmailTriageEnv()

def get_ground_truth(text: str) -> Optional[Dict[str, str]]:
    """Check if the input text matches a database entry to show ground truth analysis."""
    for entry in env.email_database:
        if text.strip() in entry['text'] or entry['text'] in text:
            return {
                "category": entry['category'],
                "priority": entry['priority'],
                "department": entry['department']
            }
    return None

def triage_analysis(text):
    """Gradio logic for email triage with performance metrics."""
    if not text or len(text.strip()) == 0:
        return "Please enter email text.", "", ""

    # 1. Model Prediction
    prediction_raw = triage_model.predict(text)
    pred_dict = prediction_raw.dict() if hasattr(prediction_raw, 'dict') else prediction_raw
    
    # 2. Performance Metrics
    truth = get_ground_truth(text)
    
    scores_html = ""
    reward_html = ""
    comparison_html = ""
    
    if truth:
        # Calculate Scores
        scores = evaluate(pred_dict, truth)
        # Calculate Reward using env logic (internally calls reward_fn)
        _, reward, _, _, info = env.step(pred_dict) # This is a bit hacky for UI but works
        breakdown = info['reward_breakdown']
        
        scores_html = f"""
        <div style='display: flex; gap: 15px; margin-top: 10px;'>
            <div style='text-align: center; background: #2d2d2d; padding: 10px; border-radius: 8px; flex: 1;'>
                <div style='font-size: 0.8em; color: #aaa;'>EASY</div>
                <div style='font-size: 1.2em; font-weight: bold; color: #4caf50;'>{scores['easy']:.2f}</div>
            </div>
            <div style='text-align: center; background: #2d2d2d; padding: 10px; border-radius: 8px; flex: 1;'>
                <div style='font-size: 0.8em; color: #aaa;'>MEDIUM</div>
                <div style='font-size: 1.2em; font-weight: bold; color: #ff9800;'>{scores['medium']:.2f}</div>
            </div>
            <div style='text-align: center; background: #2d2d2d; padding: 10px; border-radius: 8px; flex: 1;'>
                <div style='font-size: 0.8em; color: #aaa;'>HARD</div>
                <div style='font-size: 1.2em; font-weight: bold; color: #2196f3;'>{scores['hard']:.2f}</div>
            </div>
        </div>
        """
        
        reward_html = f"""
        <div style='background: #1e3a5f; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #64b5f6;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-weight: 600;'>Step Reward</span>
                <span style='font-size: 1.5em; font-weight: 900; color: #81c784;'>{reward:.2f}</span>
            </div>
            <div style='font-size: 0.8em; color: #90caf9; margin-top: 5px;'>
                Penalty: {breakdown['penalty']} | Total Raw: {breakdown['total']}
            </div>
        </div>
        """
        
        comparison_html = f"""
        <div style='margin-top: 20px; border: 1px solid #333; padding: 10px; border-radius: 8px;'>
            <h4 style='margin-top: 0;'>Benchmark Comparison</h4>
            <table style='width: 100%; font-size: 0.9em;'>
                <tr><th style='text-align: left;'>Field</th><th style='text-align: left;'>Prediction</th><th style='text-align: left;'>Truth</th></tr>
                <tr><td>Category</td><td>{pred_dict['category']}</td><td>{truth['category']}</td></tr>
                <tr><td>Priority</td><td>{pred_dict['priority']}</td><td>{truth['priority']}</td></tr>
                <tr><td>Dept</td><td>{pred_dict['department']}</td><td>{truth['department']}</td></tr>
            </table>
        </div>
        """
    else:
        reward_html = "<div style='opacity: 0.5; padding: 10px;'>Submit an example variant to see performance metrics.</div>"

    # 3. Main Triage Card
    colors = {
        'Urgent': '#f44336', 'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50',
        'Spam': '#9e9e9e', 'Inquiry': '#2196f3', 'Complaint': '#ff5722', 'Request': '#673ab7'
    }
    
    cat = pred_dict.get('category')
    pri = pred_dict.get('priority')
    dep = pred_dict.get('department')
    
    main_card = f"""
    <div style="background: #1a1a1a; padding: 20px; border-radius: 12px; border-left: 5px solid {colors.get(pri, '#fff')}; color: white; font-family: sans-serif;">
        <h3 style="margin-top: 0; color: #eee;">📧 Prediction Results</h3>
        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 5px;">
            <span style="background: #333; padding: 4px 10px; border-radius: 4px;">{cat}</span>
            <span style="background: #333; padding: 4px 10px; border-radius: 4px;">{pri}</span>
            <span style="background: #333; padding: 4px 10px; border-radius: 4px;">{dep}</span>
        </div>
    </div>
    """
    
    return main_card + scores_html + reward_html, comparison_html

def launch_app():
    with gr.Blocks(title="AI Email Triage Evaluation System", theme=gr.themes.Default(primary_hue="blue")) as demo:
        gr.Markdown("# 📧 Production-Grade AI Email Triage")
        gr.Markdown("Zero-Shot Evaluation Pipeline with Real-time Scoring and Reward Feedback.")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Email Content", 
                    placeholder="Paste email text for triage...", 
                    lines=10
                )
                submit_btn = gr.Button("🚀 Run Phase 1 Evaluation", variant="primary")
                
                gr.Examples(
                    examples=[
                        ["I want to buy 100 units of your pro plan."],
                        ["My server is melting down! Help!"],
                        ["I am unhappy with the service last week."],
                        ["Win a free cruise by clicking here!"]
                    ],
                    inputs=input_text
                )
            
            with gr.Column(scale=1):
                analysis_output = gr.HTML(label="Performance Metrics")
                comparison_output = gr.HTML(label="Truth Comparison")
        
        submit_btn.click(
            fn=triage_analysis, 
            inputs=input_text, 
            outputs=[analysis_output, comparison_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    launch_app()
