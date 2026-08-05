import json

from core.logger_config import logger
from infrastructure.ollama.client import OllamaClientInstance
from infrastructure.vector_db.client import QdrantClientInstance
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


class OllamaService:
    """
    Service class responsible for communication with an Ollama LLM server.

    This class provides high-level methods for:
    - query rewriting,
    - RAG search planning,
    - final answer generation,
    - complete retrieval and generation pipeline.

    Attributes:
        model_name (str):
            Name of the Ollama model used for inference.

        base_url (str):
            URL address of the Ollama API server.

    Example:
        service = OllamaService(
            model_name="gemma3:4b",
            base_url="http://ollama:11434"
        )

        response = service.ollamaFinalAnswer(
            "What is UWB localization?",
            context
        )
    """

    def __init__(self, ollama_client, top_k: int = 10):

        self.client = ollama_client
        self.top_k = top_k

        # =====================================
        # EMBEDDING MODEL (Ollama)
        # =====================================

        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text", base_url=self.client.base_url
        )

        # =====================================
        # QDRANT VECTOR STORE
        # =====================================

        self.vector_store = QdrantVectorStore(
            client=QdrantClientInstance, collection_name="documents"
        )

        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    def ollamaPlanner(self, user_prompt: str) -> dict:
        """
        You are a retrieval planner.

        Your task is to create a retrieval plan for a vector database.

        Always return ONLY a valid JSON object (dictionary).
        Do not include explanations, markdown, comments, code fences, or any additional text.

        Input:
        - user_prompt (str): Original user question.
        - context_data (str, optional): Retrieved context from the vector database.

        Output format:

        First call (no context):

        {
            "query_plan": [
                "query 1",
                "query 2",
                "query 3",
                "query 4",
                "query 5"
            ]
        }

        Raises:
            requests.HTTPError:
                If Ollama API request fails.
        """

        system_prompt = """
        You are the search planner of a Retrieval-Augmented Generation (RAG) system.

        You receive:
        - user_prompt

        Your task is to generate a retrieval plan for a vector database.

        Return ONLY a valid JSON object.

        Output format:

        {
            "query_plan": [
                "<query 1>",
                "<query 2>",
                "<query 3>",
                "<query 4>",
                "<query 5>"
            ]
        }

        Rules:
        - Return ONLY the JSON object.
        - Do not output markdown, explanations, comments, or additional text.
        - "query_plan" must contain exactly 5 unique search queries.
        - Each query should explore a different aspect of the user's request.
        - Queries must be optimized for semantic vector search.
        - Use concise keyword-based phrases, not full sentences.
        - Preserve important entities, technical terms, names, versions, and abbreviations from the user prompt.
        - Avoid duplicate or overly similar queries.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content = self.client.chat(messages=messages, json_mode=True, options={"num_predict": 128})

        try:
            return json.loads(content)
        except Exception:
            return {"enough": True}

    def ollamaFinalAnswer(self, user_prompt: str, context_data: str = "") -> dict | str:
        """
        Generate final answer using retrieved context.

        The model receives:
        - user question,
        - retrieved documents.

        The model must answer only using provided information.

        Args:
            user_prompt (str):
                Original user question.

            context_data (str):
                Retrieved information from vector database.

        Returns:
            dict:
                JSON answer containing generated answer.

                Example:

                {
                    "answer": "Generated answer text"
                }

        Raises:
            requests.HTTPError:
                If Ollama API request fails.
        """

        system_prompt = """
    You are an expert Retrieval-Augmented Generation (RAG) assistant.

    You receive:

    - User question
    - Context retrieved from a vector database

    Your task is to answer the user's question using ONLY the provided context.

    Instructions:

    - Read the entire context carefully.
    - Combine information from multiple fragments.
    - Ignore duplicated fragments.
    - Summarize when appropriate.
    - Use only the provided information.
    - Never invent facts.
    - If some details are missing, simply omit them.
    - Never mention the context.
    - Never mention the vector database.
    - Never explain your reasoning.

    Always produce the best possible answer from the available context.

    Even if the context is incomplete, answer using everything that is available.

    Return ONLY valid JSON is the most important requirement:
    But the answer field can be long, detailed and multi-paragraph.

    {
        "answer": "<final answer text>"
    }

    Do not return false or answer.

    Do not generate new search queries.

    No markdown.

    No explanations.

    Only JSON.
    """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"CONTEXT:\n{context_data}"},
            {"role": "user", "content": user_prompt},
        ]

        content = self.client.chat(
            messages=messages,
            json_mode=True,
            options={"num_ctx": 4096, "num_predict": 3000, "temperature": 0.7, "top_p": 0.7},
        )

        try:
            return content
        except Exception as e:
            logger.error(f"Error occurred while processing Ollama response: {e}")
            return "An error occurred while generating the response."

    def searchAndGenerate(self, user_prompt: str) -> str:

        planner_result = self.ollamaPlanner(user_prompt)

        query_plan = planner_result.get("query_plan", [user_prompt])

        logger.debug(f"Generated queries: {query_plan}")

        retriever = self.index.as_retriever(similarity_top_k=5)

        all_nodes = []

        for query in query_plan:

            logger.debug(f"Searching Qdrant: {query}")

            nodes = retriever.retrieve(query)

            all_nodes.extend(nodes)

        unique_nodes = {}

        for node in all_nodes:

            node_id = node.node.node_id

            current_score = node.score or 0

            if node_id not in unique_nodes or current_score > (unique_nodes[node_id].score or 0):

                unique_nodes[node_id] = node

        nodes = sorted(unique_nodes.values(), key=lambda x: x.score or 0, reverse=True)

        nodes = nodes[: self.top_k]

        for node in nodes[:5]:
            logger.debug(f"score={node.score}, id={node.node.node_id}")

        context = "\n\n".join(node.node.get_content() for node in nodes)

        logger.debug(f"Context size: {len(context)} chars")

        answer = self.ollamaFinalAnswer(user_prompt=user_prompt, context_data=context)
        logger.debug(f"Final answer: {answer}")
        # Model is expected to return a JSON object with a "response" field. However, if the model returns a string, we handle it gracefully.
        try:
            try:
                resulat = answer.get("response")
                return resulat
            except:
                resulat = answer.get("answer")
                return resulat
        except:
            if isinstance(answer, str):
                return answer


ollamaInit = OllamaService(ollama_client=OllamaClientInstance)
