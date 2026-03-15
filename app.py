"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from gradio.themes.utils import colors
from src.chat import Chatbot

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()
    
    def chat(message, history):
        """
        TODO:Generate a response for the current message in a Gradio chat interface.
        
        This function is called by Gradio's ChatInterface every time a user sends a message.
        You only need to generate and return the assistant's response - Gradio handles the
        chat display and history management automatically.

        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs, where each pair is
                           [user_message, assistant_message]
                           Example:
                           [
                               ["What schools offer Spanish?", "The Hernandez School..."],
                               ["Where is it located?", "The Hernandez School is in Roxbury..."]
                           ]

        Returns:
            str: The assistant's response to the current message.


        Note:
            - Gradio automatically:
                - Displays the user's message
                - Displays your returned response
                - Updates the chat history
                - Maintains the chat interface
            - You only need to:
                - Generate an appropriate response to the current message
                - Return that response as a string
        """
        return chatbot.get_response(message, history)

    
    
    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="The Planner of Beaverton",
        description="Ask me anything about the MIT course catalog! Since I am a free tier chatbot, I may give a 503 error when I'm busy. If that happens, please try again a few seconds later.",
        examples=[
            "What are some easy and introductory HASS classes to take?"
        ],
        cache_examples=True
    )
    
    return demo

def get_style():
    theme = gr.themes.Soft(
        primary_hue=colors.Color(
            name="mit_red",
            c50="#ffe6ea", 
            c100="#ffccd5", 
            c200="#ff99aa",
            c300="#ff6680",
            c400="#ff3355",
            c500="#750014",
            c600="#e60026",
            c700="#cc0022",
            c800="#b3001e",
            c900="#99001a",
            c950="#750014")
    )
    css = """.bubble-wrap.svelte-kpz1
        { background: url(https://brand.mit.edu/sites/default/files/styles/tile_narrow/public/2023-08/tim-full-body-three-quarter-view.png?itok=iWI5CwQI);
          background-position: center
        } """
    return theme, css

if __name__ == "__main__":
    demo = create_chatbot()
    theme, css = get_style()
    demo.launch(theme=theme, css=css)
