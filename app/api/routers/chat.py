from typing import Annotated

from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.ocigenai import OCIGenAIModel
from api.models.ociodsc import OCIOdscModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse
from api.setting import SUPPORTED_OCIGENAI_CHAT_MODELS, SUPPORTED_OCIODSC_CHAT_MODELS, DEFAULT_MODEL

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


@router.post("/completions", response_model=ChatResponse | ChatStreamResponse, response_model_exclude_unset=True)
async def chat_completions(
        chat_request: Annotated[
            ChatRequest,
            Body(
                examples=[
                    {
                        "model": "cohere.command-r-plus",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"},
                        ],
                    }
                ],
            ),
        ]
):
    

    model_name = chat_request.model
    
    if model_name is None:
        chat_request.model = DEFAULT_MODEL
    try:
        model_type = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["type"]
    except:
        model_type = SUPPORTED_OCIODSC_CHAT_MODELS[model_name]["type"]
    # Exception will be raised if model not supported.
    
    if model_type == "datascience":        
        model = OCIOdscModel()    # Data Science models        
    elif model_type == "ondemand":
        model = OCIGenAIModel()    # GenAI service ondemand models        
    elif model_type == "dedicated":
        model = OCIGenAIModel()    # GenAI service dedicated models

    model.validate(chat_request)
    
    if chat_request.stream:
        return StreamingResponse(
            content=model.chat_stream(chat_request), media_type="text/event-stream"
        )
    return model.chat(chat_request)
