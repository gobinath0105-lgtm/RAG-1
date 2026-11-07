import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Connect to Groq
client = Groq(api_key=api_key)


def generate_answer(query, results):

    # Prepare context from retrieved chunks
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

    # Send context and question to Groq
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