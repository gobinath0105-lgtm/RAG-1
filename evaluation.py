from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
from retriver import retrieve_chunks
evaluation_cases =[
    

    {
        "question": "What projects have I completed?",
        "expected_keywords": [
            "MNIST",
            "Cat vs Dog",
            "classification"
        ]
    },
    {
        "question": "What is my educational qualification?",
        "expected_keywords": [
            "B.Tech",
            "Artificial Intelligence",
            "Data Science"
        ]
    },
    {
        "question": "What is my work experience?",
        "expected_keywords": [
            "internship",
            "experience"
        ]
    }
]
def calculate_hit_rate(results, expected_keywords):

    for result in results:

        context = result["context"].lower()

        for keyword in expected_keywords:

            if keyword.lower() in context:
                return 1

    return 0

def evaluate_retrieval(
    evaluation_cases,
    model,
    index,
    all_chunks
):

    total_hits = 0

    for case in evaluation_cases:

        query = case["question"]
        expected = case["expected_keywords"]

        results = retrieve_chunks(
            query=query,
            model=model,
            index=index,
            all_chunks=all_chunks,
            top_k=5
        )

        hit = calculate_hit_rate(
            results,
            expected
        )

        total_hits += hit

        print("\nQuestion:", query)
        print("Hit:", hit)

    hit_rate = total_hits / len(evaluation_cases)

    print("\n========== EVALUATION RESULTS ==========")
    print("Total questions:", len(evaluation_cases))
    print("Total hits:", total_hits)
    print("Hit Rate@5:", round(hit_rate * 100, 2), "%")
# Recall@5 evaluation function
def calculate_recall_at_k(results, relevant_chunk_ids, k=5):

    retrieved_ids = set(
        result["chunk_id"] for result in results[:k]
    )

    relevant_ids = set(relevant_chunk_ids)

    if len(relevant_ids) == 0:
        return 0

    found_relevant = retrieved_ids.intersection(relevant_ids)

    recall = len(found_relevant) / len(relevant_ids)

    return recall

def check_faithfulness(context, answer):

    prompt = f"""
    Retrieved Context:
    {context}

    Generated Answer:
    {answer}

    Check whether every factual claim in the generated answer
    is supported by the retrieved context.

    Return only:
    FAITHFUL
    or
    UNFAITHFUL
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

context = """
The Cat vs Dog CNN classifier achieved 79.1% test accuracy.
"""

answer1 = "The Cat vs Dog classifier achieved 79.1% test accuracy."

answer2 = "The Cat vs Dog classifier achieved 79.1% accuracy and was trained on 50,000 images."

print("Test 1:", check_faithfulness(context, answer1))
print("Test 2:", check_faithfulness(context, answer2))
relevant_chunks = {
    "What are my machine learning skills?": [3, 4],

    "What deep learning technologies do I know?": [4],

    "What technical projects are listed?": [6, 7, 8, 9, 10],

    "What is my educational qualification?": [0, 2]
}
answer_correctness_cases = [
    {
        "question": "What is my educational qualification?",
        "expected_answer": (
            "I am a second-year B.Tech student in "
            "Artificial Intelligence and Data Science "
            "at PA College of Engineering and Technology, Pollachi."
        )
    },
    {
        "question": "What are my machine learning skills?",
        "expected_answer": (
            "Scikit-learn, Linear Regression, Logistic Regression, "
            "KNN, Decision Trees, Random Forest, SVM, Naive Bayes, "
            "K-Means, PCA, Gradient Boosting, XGBoost, "
            "Feature Engineering, Cross-Validation and Model Evaluation."
        )
    },
    {
        "question": "What deep learning technologies do I know?",
        "expected_answer": (
            "TensorFlow, Keras, Neural Networks, CNN, RNN, LSTM, "
            "Transfer Learning, Transformers and Attention Mechanism."
        )
    },
    {
        "question": "What projects have I completed?",
        "expected_answer": (
            "MNIST Handwritten Digit Classification, Cat vs Dog Image "
            "Classification, House Price Prediction and Customer Churn Prediction."
        )
    },
    {
        "question": "Do I have professional AI industry experience?",
        "expected_answer": (
            "The resume does not provide evidence of professional "
            "AI industry experience."
        )
    }
]
