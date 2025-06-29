from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import pipeline


load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

@api_view(["POST"])
def ask_question(request):
    try:
        question = request.data.get("question", "")
        if not question:
            return Response({"error": "No question provided"}, status=400)

       
        query_vector = embedder.encode(question).tolist()

        
        search_result = qdrant.search(
            collection_name="bio11",
            query_vector=query_vector,
            limit=5,  
            with_payload=True,
            score_threshold=0.5  
        )

        
        context_chunks = [hit.payload["text"] for hit in search_result if "text" in hit.payload]
        context = " ".join(context_chunks)

        print("\n--- Retrieved Context Chunks ---")
        for chunk in context_chunks:
            print(chunk)
        print("end of context\n")

        if not context:
            return Response({"answer": "No relevant content found in document."})

        
        result = qa_pipeline({
            "context": context,
            "question":  f"{question.strip()}. Explain the answer in atleast 100 words."
        })

        return Response({"answer": result["answer"]})

    except Exception as e:
        return Response({"error": str(e)}, status=500)
