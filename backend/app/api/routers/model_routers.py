from fastapi import APIRouter
from core.logger_config import logger
from services.ragServices import ollamaInit
from api.schemas.queryModels import TextRequest, TextResponse


router_modelOLLAMA = APIRouter(
    prefix="/ollama-event",
    tags=["Ollama"]
)


@router_modelOLLAMA.post(
    "/query",
    response_model=TextResponse,
    summary="Generate text response using Ollama model",
    description="""
## Ollama Text Generation API

This endpoint sends a text query to the Ollama language model
and returns a generated response.

### Features:
- Local LLM text generation
- Configurable generation parameters
- Optional system prompt support

### Parameters:
- **text** - User input prompt.

### Example use cases:
- AI assistant
- Local chatbot
- RAG question answering
- Document analysis

### Workflow:
1. Client sends text request.
2. Backend forwards query to Ollama.
3. Ollama generates response.
4. API returns generated text.
"""
)
def query(req: TextRequest) -> TextResponse:
    """
    Execute Ollama text generation.

    Receives user prompt with optional generation parameters
    and returns generated model output.

    Args:
        req:
            TextRequest containing input text and generation settings.

    Returns:
        TextResponse:
            Generated text from Ollama model.

    Raises:
        Exception:
            Returns error message if generation fails.
    """

    try:
        logger.debug(f"Received Ollama request: {req}")

        response = ollamaInit.searchAndGenerate(req.text)
        logger.debug(f"Ollama response: {response}")
        if type(response) != str:
            logger.error(f"Unexpected response type from Ollama: {type(response)}")
            raise ValueError(f"Unexpected response type from Ollama: {type(response)}")
        
        return TextResponse(
            text=response,
            model_name="ollama"
        )

    except Exception as e:
        logger.error(f"❌ Error handling Ollama request: {e}")

        return TextResponse(
            text="Error occurred while processing the request"
        )