import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# -------------------------------------------------------------
# 1. INITIALIZATION & SETUP
# -------------------------------------------------------------
# ⚠️ MAKE SURE YOU PUT YOUR ACTUAL GROQ API KEY HERE!
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("Initializing AI Models...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
qa_chain = None 

# -------------------------------------------------------------
# 2. BACKEND LOGIC
# -------------------------------------------------------------
def process_uploaded_file(file_path):
    global qa_chain
    if file_path is None:
        return "⚠️ Please upload a valid PDF file."
    
    try:
        loader = PyPDFLoader(file_path)
        document = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        texts = text_splitter.split_documents(document)
        
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return "✅ PDF processed successfully! You can now chat with your document."
    
    except Exception as e:
        return f"❌ Unable to process the PDF.\n\nDetails: {str(e)}"

def predict(message, history):
    global qa_chain
    
    clean_history = []
    for msg in (history or []):
        if isinstance(msg, dict) and "role" in msg:
            clean_history.append(msg)
            
    if qa_chain is None:
        error_msg = "⚠️ STOP: Please upload a PDF document first!"
        clean_history.append({"role": "user", "content": message})
        clean_history.append({"role": "assistant", "content": error_msg})
        return "", clean_history, error_msg

    if not message.strip():
        return "", clean_history, "No query provided."

    try:
        response = qa_chain.invoke({"query": message})
        answer = response["result"]
        source_docs = response.get("source_documents", [])
        
        sources_text = ""
        for i, doc in enumerate(source_docs):
            sources_text += f"--- SOURCE CHUNK {i+1} ---\n{doc.page_content.strip()}\n\n"
        
        if not sources_text:
            sources_text = "No specific chunks were retrieved for this query."

        clean_history.append({"role": "user", "content": message})
        clean_history.append({"role": "assistant", "content": answer})
        
        return "", clean_history, sources_text

    except Exception as e:
        error_msg = f"❌ SYSTEM CRASH: {str(e)}"
        clean_history.append({"role": "user", "content": message})
        clean_history.append({"role": "assistant", "content": error_msg})
        return "", clean_history, "Check the chat window for the specific error."

# -------------------------------------------------------------
# 3. GRADIO INTERFACE DESIGN
# -------------------------------------------------------------
with gr.Blocks() as demo:
    
    # --- NEW: Header Section with Dark Mode Button ---
    with gr.Row(equal_height=True):
        gr.Markdown("""
        # 📄 SmartDoc AI

        ### AI-powered Document Intelligence Platform

        Ask questions, summarize PDFs, and retrieve contextual information using Retrieval-Augmented Generation (RAG).
        """)
        theme_btn = gr.Button("🌙 Theme", scale=1)
    
    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="📂 Upload PDF",file_types=[".pdf"],type="filepath")
            upload_status = gr.Textbox(label="📢 Status",value="Upload a PDF to begin chatting.",interactive=False)
            
    file_upload.upload(fn=process_uploaded_file, inputs=[file_upload], outputs=[upload_status])
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="🤖 AI Assistant")
            msg = gr.Textbox(label="💬 Ask Anything",placeholder="Example: Summarize this PDF",lines=1)
            gr.Examples(
                examples=[
                    ["Summarize this document"],
                    ["What are the key findings?"],
                    ["Explain this PDF in simple words"],
                    ["List all important dates"],
                    ["Generate interview questions from this PDF"]
                ],
                inputs=msg
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Ask AI", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot], variant="secondary")
            
            with gr.Accordion("🔍 Inspect Retrieved Context", open=False):
                sources_output = gr.Textbox(label="Context injected into LLM:", lines=10, interactive=False)

    # --- Button Event Listeners ---
    submit_btn.click(predict, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_output])
    msg.submit(predict, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_output])
    
    # --- NEW: JavaScript function to toggle the theme ---
    theme_btn.click(
        None, 
        None, 
        None, 
        js="""
        function() {
            document.body.classList.toggle('dark');
        }
        """
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )