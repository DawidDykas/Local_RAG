from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """
    Request model for Ollama text queries.
    Contains the input text and optional parameters for generation control.
    """

    text: str = Field(
        ...,
        description="Input text to be processed by the Ollama model",
        example="What is the capital of France?",
        min_length=1,
        max_length=10000,
    )


class TextResponse(BaseModel):
    """
    Response model for Ollama text queries.
    Contains the generated text response.
    """

    text: str = Field(
        ...,
        description="Generated response from the Ollama model",
        example="The capital of France is Paris.",
    )
    tokens_used: int | None = Field(
        default=None, description="Number of tokens used in the generation", example=45
    )
    model_name: str | None = Field(
        default=None,
        description="Name of the Ollama model used for generation",
        example="llama2:7b",
    )
