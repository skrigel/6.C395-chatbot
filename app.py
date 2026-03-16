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

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
