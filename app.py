import gradio as gr
import time
import os
import openai
openai.api_key = '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'
# Function to handle fine-tuning training process
def train_model(file, model, model_url, finetune):
    if not file:
        return "Please upload a dataset.", gr.update(interactive=False), gr.update(interactive=False), None
    
    steps = [
        "Processing the data...",
        "Converting unstructured to structured format...",
        "Loading data and fine-tuning...",
        "Exporting model...",
        "Training finished."
    ]
    
    for step in steps:
        yield step, gr.update(interactive=False), gr.update(interactive=False), None
        time.sleep(2)  # 2-second delay for testing purposes
    
    # Simulate saving the trained model
    model_path = "granite-3.1-2b-base-finetuned.zip"
    with open(model_path, "w") as f:
        f.write("Trained model data")
    
    # After training is finished, enable the buttons and provide the model file path
    yield "Training finished.", gr.update(interactive=True), gr.update(interactive=True), model_path

# Function to handle RAG training process
def train_rag_model(file, model, model_url, embedding):
    if not file:
        return "Please upload a dataset.", gr.update(interactive=False), gr.update(interactive=False), None
    
    steps = [
        "Processing the data...",
        "Converting unstructured to structured format...",
        "Loading data and training RAG model...",
        "Exporting model...",
        "Training finished."
    ]
    
    for step in steps:
        yield step, gr.update(interactive=False), gr.update(interactive=False), None
        time.sleep(2)  # 2-second delay for testing purposes
    
    # Simulate saving the trained model
    model_path = "granite-3.1-2b-base-finetuned.zip"
    with open(model_path, "w") as f:
        f.write("Trained model data")
    
    # After training is finished, enable the buttons and provide the model file path
    yield "Training finished.", gr.update(interactive=True), gr.update(interactive=True), model_path

# Define the chat function to be used by the chat interface
def respond(message, history):
    # Engineered system prompt for a banking chatbot specialized in tasks for PISTA.
    system_prompt = (
        "You are a highly specialized banking chatbot for PISTA, a fictional bank. "
        "You are trained on the following dummy user data:\n"
        "1. John Doe - Account number: 123456789, Balance: $10,000\n"
        "2. Jane Smith - Account number: 987654321, Balance: $25,000\n"
        "3. Robert Brown - Account number: 112233445, Balance: $5,000\n"
        "4. Alice White - Account number: 556677889, Balance: $15,500\n"
        "5. Michael Johnson - Account number: 998877665, Balance: $8,750\n"
        "6. Emily Davis - Account number: 443322110, Balance: $12,300\n"
        "Your role is to help users with banking tasks such as account inquiries, transactions, loans, and credit information. "
        "If a user asks a question outside the banking domain, respond with: "
        "'I'm sorry, I can only assist with banking-related questions.' "
        "Ensure your responses are professional, clear, and helpful."
    )
    # Build conversation history with the system prompt at the beginning.
    conversation = [{"role": "system", "content": system_prompt}]
    
    # Append any previous conversation.
    for user_msg, bot_msg in history or []:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": bot_msg})
    
    # Append the latest user message.
    conversation.append({"role": "user", "content": message})
    
    # Generate a response from the OpenAI API.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    
    reply = response.choices[0].message['content']
    
    # Initialize history if it doesn't exist.
    history = history or []
    history.append((message, reply))
    
    # Clear the textbox after submission.
    return "", history


# Gradio interface
with gr.Blocks(theme=gr.themes.Base(), css="body { background-color: black; color: white; }") as demo:
    # Main Page
    with gr.Column(visible=True) as main_page:
        gr.Markdown("<h1>OCT4LLM</h1>")
        gr.Markdown("<p>Select an option below to proceed:</p>")
        with gr.Row():
            finetune_btn = gr.Button("Fine-tune Model")
            # rag_btn = gr.Button("RAG on Model")
    
    # Fine-tune Page
    with gr.Column(visible=False) as finetune_page:
        gr.Markdown("<h1>Fine-tune Model</h1>")
        gr.Markdown("<p>Upload a dataset, select a model, or provide a model URL for fine-tuning.</p>")
        upload = gr.File(label="Upload Dataset")
        model_url_input = gr.Textbox(label="Paste Model URL (Optional)")
        model_select = gr.Dropdown(
            ["ibm-granite/granite-3.1-2b-base", "ibm-granite/granite-3.1-8b-instruct", "ibm-granite/granite-3.1-1b-a400m-base"],
            label="Select Model",
            interactive=True
        )
        finetune_select = gr.Dropdown(
            ["Lora", "Q-lora", "Knowledge Distillation"],
            label="Select Fine-tune Method",
            interactive=True
        )
        train_btn = gr.Button("Train Model")
        train_output = gr.Textbox(label="Training Status", interactive=False)
        with gr.Row():
            download_btn = gr.File(label="Download Trained Model", interactive=False)
            deploy_btn = gr.Button("Deploy Model", interactive=False)
        back_finetune = gr.Button("Back to Main")

    # RAG Page
    with gr.Column(visible=False) as rag_page:
        gr.Markdown("<h1>RAG Model</h1>")
        gr.Markdown("<p>Coming Soon...........</p>")
        # upload_rag = gr.File(label="Upload Dataset")
        # model_url_input_rag = gr.Textbox(label="Paste Model URL (Optional)")
        # model_select_rag = gr.Dropdown(
        #     ["Item 1", "Item 2", "Item 3"],
        #     label="Select Model",
        #     interactive=True
        # )
        # embedding_select = gr.Dropdown(
        #     ["Item 1", "Item 2", "Item 3"],
        #     label="Select Embeddings",
        #     interactive=True
        # )
        # train_rag_btn = gr.Button("Train Model")
        # train_rag_output = gr.Textbox(label="Training Status", interactive=False)
        # with gr.Row():
        #     download_rag_btn = gr.File(label="Download Trained Model", interactive=False)
        #     deploy_rag_btn = gr.Button("Deploy Model", interactive=False)
        back_rag = gr.Button("Back to Main")
    
    # Chat Page
    with gr.Column(visible=False) as chat_page:
        gr.Markdown("## OpenAI Chatbot")
        chatbot = gr.Chatbot(label="Chatbot")
        msg = gr.Textbox(placeholder="Type your message here...", label="Message")
        # Connect the chat function so that when a message is submitted, a response is generated
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # Button Actions
    finetune_btn.click(
        lambda: (
            gr.update(visible=True),  # Show finetune_page
            gr.update(visible=False), # Hide rag_page
            gr.update(visible=False), # Hide main_page
            gr.update(visible=False)  # Hide chat_page
        ),
        outputs=[finetune_page, rag_page, main_page, chat_page]
    )
    rag_btn.click(
        lambda: (
            gr.update(visible=False), # Hide finetune_page
            gr.update(visible=True),  # Show rag_page
            gr.update(visible=False), # Hide main_page
            gr.update(visible(False)) # Hide chat_page (alternate syntax)
        ),
        outputs=[finetune_page, rag_page, main_page, chat_page]
    )
    back_finetune.click(
        lambda: (
            gr.update(visible=False), # Hide finetune_page
            gr.update(visible(False)),# Hide rag_page
            gr.update(visible(True)), # Show main_page
            gr.update(visible(False)) # Hide chat_page
        ),
        outputs=[finetune_page, rag_page, main_page, chat_page]
    )
    # back_rag.click(
    #     lambda: (
    #         gr.update(visible(False)), # Hide finetune_page
    #         gr.update(visible(False)), # Hide rag_page
    #         gr.update(visible(True)),  # Show main_page
    #         gr.update(visible(False))  # Hide chat_page
    #     ),
    #     outputs=[finetune_page, rag_page, main_page, chat_page]
    # )

    # Training actions
    train_btn.click(
        train_model,
        inputs=[upload, model_select, model_url_input, finetune_select],
        outputs=[train_output, download_btn, deploy_btn, download_btn]
    )
    train_rag_btn.click(
        train_rag_model,
        inputs=[upload_rag, model_select_rag, model_url_input_rag, embedding_select],
        outputs=[train_rag_output, download_rag_btn, deploy_rag_btn, download_rag_btn]
    )

    # Deploy action: switch to the chat page
    deploy_btn.click(
        lambda: (
            gr.update(visible=False),  # Hide finetune_page
            gr.update(visible=False),  # Hide rag_page
            gr.update(visible=False),  # Hide main_page
            gr.update(visible=True)    # Show chat_page
        ),
        outputs=[finetune_page, rag_page, main_page, chat_page]
    )

demo.queue().launch(share=True)
