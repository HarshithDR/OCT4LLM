import gradio as gr

# Functions for handling page transitions
def show_finetune_page():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def show_rag_page():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def show_main_page():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

# Handling selection change
def handle_selection(selected_option):
    return f"Selected: {selected_option}"

# Train model function
def train_model(file, model, model_url, finetune):
    if not file:
        return "Please upload a dataset."
    if model_url:
        return f"Training started with custom model from {model_url} and fine-tune option {finetune}."
    return f"Training started with selected model {model} and fine-tune option {finetune}."

# Train RAG model function
def train_rag_model(file, model, model_url, embedding):
    if not file:
        return "Please upload a dataset."
    if model_url:
        return f"Training RAG model with custom model from {model_url} and embedding option {embedding}."
    return f"Training RAG model with selected model {model} and embedding option {embedding}."


css = """
/* General Styles */

body {
    font-family: 'Arial', sans-serif;
    background-color: #121212; /* Dark background */
    color: #ffffff; /* White text */
    margin: 0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh; 
}

/* Dark Mode Container */
.gradio-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    width: 90%;
    max-width: 1000px;
    padding: 30px;
    background: #1e1e1e; /* Darker card background */
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.1);
}

/* Header Styling */
h1 {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 10px;
    color: #ff9800; /* Accent color */
}

p {
    font-size: 1.1rem;
    color: #bdbdbd; /* Lighter gray for better readability */
}

/* Buttons */
.custom-btn {
    background-color: #ff9800;
    color: #ffffff;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    margin: 10px;
}

.custom-btn:hover {
    background-color: #e68900;
    transform: scale(1.05);
}

/* Form Elements - Dark Mode */
input, select, textarea {
    width: 100%;
    padding: 10px;
    margin-top: 8px;
    background: #2c2c2c;
    border: 1px solid #555;
    color: #fff;
    border-radius: 5px;
}





.gradio-container .file-upload:hover {
    border-color: #ff9800;
}

/* Layout Adjustments */
.gradio-container .row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 15px;
}

.gradio-container .column {
    flex: 1;
    min-width: 280px;
    padding: 10px;
}

/* Dark Mode for Dropdowns */
.gradio-container select {
    background: #2c2c2c;
    color: #ffffff;
    border: 1px solid #777;
}

/* Responsive Design */
@media (max-width: 768px) {
    .gradio-container {
        width: 95%;
        padding: 20px;
    }

    h1 {
        font-size: 2rem;
    }

    .custom-btn {
        font-size: 14px;
        padding: 10px 16px;
    }
}

/* Fine-Tune Page Padding */
#finetune-page {
    padding-top: 800px;
}

/* RAG Page Padding */
#rag-page {
    padding-top: 500px;
}



"""




with gr.Blocks(theme=gr.themes.Base(), css=css) as demo:
    # Main Page
    with gr.Column(visible=True) as main_page:
        gr.Markdown("<h1>OCT4LLM</h1>")
        gr.Markdown("<p>Beyond your Imagination.</p>")
        gr.Markdown("<p>Talk With your Bot</p>")
        with gr.Row():
            finetune_btn = gr.Button("Finetune Model", elem_classes=["custom-btn"])
            rag_btn = gr.Button("RAG on Model", elem_classes=["custom-btn"])

    # Fine-tune Page
    with gr.Column(visible=False, elem_id="finetune-page") as finetune_page:
        gr.Markdown("<h1>Fine-tune Model</h1>")
        gr.Markdown("<p>Upload dataset, select model, or provide a model URL for fine-tuning.</p>")

        with gr.Row():
            upload = gr.File(label="Upload Dataset")
            with gr.Column():
                radio_btn = gr.Radio(["Structured", "Unstructured"], label="File Method")
                output = gr.Textbox(label="Your Selection")
                radio_btn.change(handle_selection, inputs=radio_btn, outputs=output)

        model_url_input = gr.Textbox(label="Choose a LLM Model")
        model_select = gr.Dropdown(["Item 1", "Item 2", "Item 3"], label="Top Models", interactive=True)

        gr.Markdown("**Select Fine-tune Method**")
        with gr.Row():
            finetune_select = gr.Dropdown(["Item 1", "Item 2", "Item 3"], interactive=True)
            advanced_btn = gr.Button("Advanced", size="sm")
        train_btn = gr.Button("Train Model", elem_classes=["custom-btn"])

        with gr.Row():
            download_btn = gr.Button("Download Trained Model", elem_classes=["custom-btn"])
            deploy_btn = gr.Button("Deploy Model", elem_classes=["custom-btn"])

        back_finetune = gr.Button("Back to Main", elem_classes=["custom-btn"])

    # RAG Page
    with gr.Column(visible=False,  elem_id="rag-page") as rag_page:
        gr.Markdown("<h1>RAG Model</h1>")
        gr.Markdown("<p>Upload dataset, select model, or provide a model URL along with embeddings.</p>")
        
        upload_rag = gr.File(label="Upload Dataset")
        model_url_input_rag = gr.Textbox(label="Paste Model URL (Optional)")
        model_select_rag = gr.Dropdown(["Item 1", "Item 2", "Item 3"], label="Select Model", interactive=True)
        embedding_select = gr.Dropdown(["Item 1", "Item 2", "Item 3"], label="Select Embeddings", interactive=True)

        train_rag_btn = gr.Button("Train Model", elem_classes=["custom-btn"])
        train_rag_output = gr.Textbox(label="Training Status", interactive=False)

        with gr.Row():
            download_rag_btn = gr.Button("Download Trained Model", elem_classes=["custom-btn"])
            deploy_rag_btn = gr.Button("Deploy Model", elem_classes=["custom-btn"])

        back_rag = gr.Button("Back to Main", elem_classes=["custom-btn"])

    # Button Actions
    finetune_btn.click(show_finetune_page, outputs=[finetune_page, rag_page, main_page])
    rag_btn.click(show_rag_page, outputs=[finetune_page, rag_page, main_page])
    back_finetune.click(show_main_page, outputs=[finetune_page, rag_page, main_page])
    back_rag.click(show_main_page, outputs=[finetune_page, rag_page, main_page])

    # Training actions
    train_rag_btn.click(train_rag_model, inputs=[upload_rag, model_select_rag, model_url_input_rag, embedding_select], outputs=train_rag_output)

demo.launch(share=True)