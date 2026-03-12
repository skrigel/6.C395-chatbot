from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from typing import List, Dict
import requests
import xml.etree.ElementTree as ET
import json

CATALOG_URL = "https://catalog.mit.edu/ribbit/index.cgi?page=getcourse.rjs&code="


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_course",
            "description": "Fetch information about a particular course from the MIT course catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_number": {
                        "type": "string", 
                        "description": "Number used to identify class (ie 6.1010)"
                    },
                },
                "required": ["course_number"],
            },
        },
    },
]

def get_course(course_number):
    url = CATALOG_URL + course_number
    r = requests.get(url)

    outer = ET.fromstring(r.text)

    # The catalog wraps HTML in CDATA inside a <course> element
    course_elem = outer.find(".//course")
    if course_elem is None or not course_elem.text:
        return {"course": course_number, "error": "Course not found"}

    # Re-parse the inner HTML as XML
    inner = ET.fromstring(f"<root>{course_elem.text}</root>")

    title = inner.findtext(".//p[@class='courseblocktitle']")
    desc = inner.findtext(".//p[@class='courseblockdesc']")

    prereqs: List[str] = []
    prereq_elem = inner.find(".//span[@class='courseblockprereq']")

    if prereq_elem is not None:
        for a in prereq_elem.findall(".//a"):
            if a.text:
                prereqs.append(a.text.strip())

    return {
        "course": course_number,
        "title": title,
        "description": desc,
        "prereqs": prereqs
    }

class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.MAX_TOKENS = 1024

    
    def format_messages(self, prompt: str, history: List[Dict]) -> List[Dict]:
        messages = [{"role": "system", "content": "You are a helpful assistant named Sendhil that specializes in helping students navigate the MIT course catalog. Please be sure to introduce yourself as an icon at the start of each response. When the user asks about a specific course, you MUST use the get_course tool to fetch accurate information. Do not make up or guess course details."}]
        for msg in history:
            content = msg["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            messages.append({"role": msg["role"], "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def format_prompt(self, user_input: str)->str:
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question

        Returns:
            str: A formatted prompt ready for the model
        
        Example prompt format:
            "You are a helpful assistant that specializes in...
             User: {user_input}
             Assistant:"

        """
 
        placeholder_prompt = f"""
        User: {user_input}
        """
        return placeholder_prompt
        
    def get_response(self, user_input, history=[])->str:
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """

        messages = self.format_messages(user_input, history)
        response = self.client.chat_completion(messages=messages, max_tokens=self.MAX_TOKENS,tools=tools,tool_choice='auto')  # type: ignore
        response_message = response.choices[0].message

        # Check if model wants to call functions
        if response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "get_course":
                    result = get_course(function_args["course_number"])

                    print("tool call result ", result)
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result),
                    })

            # Get final response with function results
            final_response = self.client.chat_completion(
                messages=messages, max_tokens=self.MAX_TOKENS
            )

            message_content = final_response.choices[0].message.content 
            return message_content if message_content else "Sorry!"
        else:
            
            return response_message.content if response_message.content else "Sorry!"
