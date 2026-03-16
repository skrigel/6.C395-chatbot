from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from typing import List, Dict
import re
import requests
import xml.etree.ElementTree as ET
import json
from .rag_chunking import PineconeService, embeddings
from sklearn.metrics.pairwise import cosine_similarity

# with open("data/s25_names.txt") as f:
#     SPRING_CLASSES = f.read()

N_RAG_CHUNKS=10

with open("src/system_prompt_rag.txt") as f:
    SYSTEM_PROMPT = f.read() #+ SPRING_CLASSES

pinecone_service = PineconeService('class-catalog-full')

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

        
    def format_prompt(self, user_input, history=[]):
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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            content = msg["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            messages.append({"role": msg["role"], "content": content})
        messages.append({"role": "user", "content": user_input})
        return messages

    
    def check_for_filters(self, user_embedding):
        user_query_filters = {}

        filters = [('is a HASS class', 'Class HASS Categories Satisfied', '$in', ["('', 'H')", "('', 'S')", "('', 'A')"]),
                    ('is a HASS-H', 'Class HASS Categories Satisfied', '$eq',  "('', 'H')"),
                    ('is a HASS-S', 'Class HASS Categories Satisfied', '$eq',  "('', 'S')"),
                    ('is a HASS-A', 'Class HASS Categories Satisfied', '$eq',  "('', 'A')"),
                    ('is a CI-H', 'Class CI-H Status', '$eq', "('', 'C, I, -, H')"),
                    ('is a CI-HW', 'Class CI-HW Status', '$eq', "('', 'C, I, -, H, W')"),
                    ('is a GIR', 'Class GIR Categories Satisfied', '$in', ["('', 'P, H, Y, 1')", "('', 'P, H, Y, 2')", "('', 'B, I, O, L')", "('', 'C, H, E, M')", "('', 'C, A, L, 1')", "('', 'C, A, L, 2')"]),
                    ('is offered in the fall term', 'Terms Offered', '$in', ['FA', 'FA, SP', 'SP, FA']),
                    ('is offered in the spring term', 'Terms Offered', '$in', ['SP', 'FA, SP', 'SP, FA']),
                    ('has a final', 'Class Has Final', '$eq', 'True'),
                    ('does not have a final', 'Class Has Final', '$eq', 'False')
                    ]
        
        similarities = []
        
        for i, (filter_phrase, filter_subject, filter_operator, filter_value) in enumerate(filters):
            filter_embedding = embeddings.embed_query(filter_phrase)
            similarities.append(cosine_similarity([user_embedding], [filter_embedding])[0][0])

        print(similarities)
        max_value = max(similarities) 
        max_index = similarities.index(max_value) 

        if max_value > 0.88:
            user_query_filters[filters[max_index][1]] = {filters[max_index][2]: filters[max_index][3]}

        return user_query_filters
        

    def get_response(self, user_input, history=[]):
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

        # Check if the user mentioned specific class numbers (e.g., 6.1210, 18.06, 21L.611)
        class_number_pattern = r'\b(\d+[A-Z]?\.[\w]+)\b'
        mentioned_classes = re.findall(class_number_pattern, user_input, re.IGNORECASE)

        # Direct ID lookup for mentioned class numbers
        direct_results = []
        if mentioned_classes:
            try:
                fetch_response = pinecone_service.index.fetch(
                    ids=mentioned_classes, namespace='course-catalog'
                )
                for vec_id, vec in fetch_response.vectors.items():
                    m = vec.metadata.copy()
                    m['score'] = 1.0  # exact match
                    direct_results.append(m)
            except Exception as e:
                print(f"Direct fetch error: {e}")

        user_embedding = embeddings.embed_query(user_input)
        user_query_filters = self.check_for_filters(user_embedding)

        rag_results = pinecone_service.query_and_filter(query_text=user_input
                                                        , filter=user_query_filters if user_query_filters else None
                                                        , top_k=N_RAG_CHUNKS, namespace='course-catalog')

        # Merge: direct results first, then semantic results (deduplicated)
        seen_ids = {r.get('Class Number') for r in direct_results}
        for r in rag_results:
            if r.get('Class Number') not in seen_ids:
                direct_results.append(r)
                seen_ids.add(r.get('Class Number'))
        rag_results = direct_results
        print(rag_results)
        rag_context = "\n".join(
            "\n".join(f"  {k}: {v}" for k, v in r.items() if k != 'score')
            + "\n"
            for r in rag_results
        )

        messages = self.format_prompt(user_input, history)
        messages.append({'role':'system'
                         , 'content': f"Here are {len(rag_results)} classes retrieved from the course catalog that may be relevant:\n{rag_context}\nIMPORTANT: ONLY reference classes, professors, and details that appear in the data above. Do NOT make up or guess any class names, numbers, or instructor names. If the information the user is asking about is not present in this data, clearly tell them it is not available in your catalog and suggest they check the MIT Course Catalog or their departmental advisor."})

        response = self.client.chat_completion(messages=messages, max_tokens=1024)
        if not response.choices[0].message.content:
            return "Sorry! Please try again :("
        return response.choices[0].message.content