# WebAnalyzer with Groq Chatbot

This is the refactored WebAnalyzer that uses Groq's Llama-3 70B model for the chatbot and mentions analysis, replacing the previous Azure OpenAI RAG implementation.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    - Add your Groq API key to `.env`:
        ```
        GROQ_API_KEY=gsk_...
        ```

3.  **Run the Application:**
    ```bash
    python main.py
    ```

## Changes from Original
- **Chatbot**: Now uses `groq` library with `llama-3.3-70b-versatile` model.
- **Mentions**: Now uses `groq` instead of Azure OpenAI.
- **Removed**: Azure OpenAI, LangChain, and ChromaDB dependencies.

