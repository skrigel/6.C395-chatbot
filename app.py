"""
Gradio Web Interface for MIT Course Chatbot
"""

import gradio as gr
from src.chat import Chatbot

def create_chatbot():
    chatbot = Chatbot()

    def chat(message, history):
        return chatbot.get_response(message, history)

    demo = gr.ChatInterface(
        chat,
        title="The (Course) Planner of Beaverton",
        description="Your MIT course catalog assistant. Ask about classes, schedules, requirements, and more.",
        examples=[
            "What are some easy and introductory HASS classes to take?",
            "What are some advanced classes in course 22?",
            "Can you suggest a schedule for this spring without any classes before 11 AM?",
            "Who teaches 6.1210?",
        ],
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
