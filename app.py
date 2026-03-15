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
        title="The (Course) Planner of Beaverton",
        description="Ask me anything about the MIT course catalog! Since I am a free tier chatbot, I may give a 503 error when I'm busy. If that happens, please try again a few seconds later. (Art credit: Margaret Zheng)",
        examples=[
            "What are some easy and introductory HASS classes to take?",
            "What are some advanced classes in course 22?",
            "Can you suggest a schedule for this spring without any classes before 11 AM?"
        ]
    )
    
    return demo

def get_style():
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="sky",
        neutral_hue="zinc"
    )
    css = """.bubble-wrap.svelte-kpz1
        { 
          background: url(https://i.pinimg.com/736x/6a/70/3a/6a703a8f55e50523d98cd1ac88dba19a.jpg);
          background-position: center;
          background-size: cover
        } """
    return theme, css

if __name__ == "__main__":
    demo = create_chatbot()
    theme, css = get_style()
    demo.launch(theme=theme, css=css)
