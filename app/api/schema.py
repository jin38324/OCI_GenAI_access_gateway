from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Iterable, Union
from typing_extensions import TypeAlias

from openai.types.shared_params.metadata import Metadata
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.chat import (
    # ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,    
)

from openai.types.chat.chat_completion import ChatCompletion as ChatResponse
from openai.resources.chat.completions import CompletionsWithStreamingResponse as ChatStreamResponse

from openai._types import SequenceNotStr
from openai.types.embedding_create_params import EmbeddingCreateParams as EmbeddingsRequest
from openai.types import CreateEmbeddingResponse as EmbeddingsResponse

# from openai.resources.chat.completions import CompletionsWithStreamingResponse
from openai.types import Model


class Models(BaseModel):
    object: str | None = "list"
    data: List[Model]


class RerankRequest(BaseModel):
    """Cohere v2-compatible rerank request."""

    model: Optional[str] = Field(default=None, min_length=1)
    query: str = Field(min_length=1)
    documents: List[str] = Field(min_length=1)
    top_n: Optional[int] = Field(default=None, ge=1)
    max_tokens_per_doc: Optional[int] = Field(default=4096, ge=1)
    priority: Optional[int] = Field(default=None, ge=0, le=999)


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankApiVersion(BaseModel):
    version: str = "2"
    is_experimental: bool = False


class RerankMeta(BaseModel):
    api_version: RerankApiVersion = Field(default_factory=RerankApiVersion)


class RerankResponse(BaseModel):
    """Cohere v2-compatible rerank response."""

    id: str
    results: List[RerankResult]
    meta: RerankMeta = Field(default_factory=RerankMeta)


class RerankResultsResponse(BaseModel):
    """Minimal /v1/rerank response used by external RAG clients."""

    results: List[RerankResult]

from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam


from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_message_function_tool_call_param import ChatCompletionMessageFunctionToolCallParam



ChatCompletionAssistantMessageParam.__annotations__["tool_calls"] = Optional[List[ChatCompletionMessageFunctionToolCallParam]]
setattr(ChatCompletionAssistantMessageParam, "tool_calls", None)

ChatCompletionUserMessageParam.__annotations__["content"] = Union[str, List[ChatCompletionContentPartParam]]
setattr(ChatCompletionUserMessageParam, "content", None)


class ChatRequest(BaseModel):
    # compatibility with OCI       
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    metadata: Optional[Metadata] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None        
    presence_penalty: Optional[float] = None    
    seed: Optional[int] = None    
    stop: Union[Optional[str], SequenceNotStr[str], None] = None   
    temperature: Optional[float] = None    
    top_p: Optional[float] = None    

    # Need mapping
    model: str
    n: Optional[int] = None    
    logprobs: Optional[bool] = None            
    parallel_tool_calls: Optional[bool] = None  # False不行，会导致grok tool call 返回空值   
    stream: Optional[Literal[False,True]] = False


    # Support but NOT compatibility with OCI
    messages: List[ChatCompletionMessageParam]
    prediction: Optional[ChatCompletionPredictionContentParam] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    response_format: Optional[completion_create_params.ResponseFormat] = None 
    stream_options: Optional[ChatCompletionStreamOptionsParam] = None
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    tools: Optional[List[ChatCompletionToolUnionParam]] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    
    # extra parameters
    extra_body: Optional[Dict] = None


    ### NOT supported by OCI
    # audio: Optional[ChatCompletionAudioParam] = None
    # modalities: Optional[List[Literal["text", "audio"]]] = None 
    # prompt_cache_key: Optional[str] = None    
    # safety_identifier: Optional[str] = None    
    # service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None    
    # store: Optional[bool] = None   
    # top_logprobs: Optional[int]    
    # user: Optional[str]    
    # web_search_options: Optional[completion_create_params.WebSearchOptions]
