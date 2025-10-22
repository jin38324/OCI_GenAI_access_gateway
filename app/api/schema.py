from typing import Dict, List, Literal, Optional, Union

from openai._types import SequenceNotStr

# from openai.resources.chat.completions import CompletionsWithStreamingResponse
from openai.types import Model
from openai.types.chat import (
    # ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata
from pydantic import BaseModel


class Models(BaseModel):
    object: str | None = "list"
    data: List[Model]


ChatCompletionAssistantMessageParam.__annotations__["tool_calls"] = Optional[
    List[ChatCompletionMessageFunctionToolCallParam]
]
setattr(ChatCompletionAssistantMessageParam, "tool_calls", None)

ChatCompletionUserMessageParam.__annotations__["content"] = Union[
    str, List[ChatCompletionContentPartParam]
]
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
    parallel_tool_calls: bool = False
    stream: Optional[Literal[False, True]] = False

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
