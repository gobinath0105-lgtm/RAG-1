
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# =====================================
# 1. LOAD API KEY
# =====================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# If local .env key is unavailable,
# get the key from Streamlit Cloud Secrets

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise ValueError("GROQ_API_KEY is not configured.")


# =====================================
# 2. CONNECT TO GROQ
# =====================================

client = Groq(api_key=api_key)


# =====================================
# 3. GENERATE ANSWER
# =====================================

def generate_answer(query, results):

    context = ""

    for i, result in enumerate(results):

        page = result["metadata"]["page"]
        source = result["metadata"]["source"]
        text = result["context"]

        context += f"""
        Chunk {i + 1}
        Source: {source}
        Page: {page}
        Content: {text}
        """

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful PDF question-answering assistant.

                Answer using ONLY the provided context.

                If the answer is not in the context, say:
                "I could not find this information in the document."

                Do not invent information.
                Mention the source page number in your answer.
                """
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}

                Question:
                {query}
                """
            }
        ],

        temperature=0.2,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content
