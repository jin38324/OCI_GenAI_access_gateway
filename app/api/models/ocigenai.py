import base64
import json
import logging
import re
import time
from abc import ABC
from typing import AsyncIterable, Iterable, Literal

import oci
from api.setting import DEBUG
from api.setting import CLIENT_KWARGS, \
    INFERENCE_ENDPOINT_TEMPLATE, \
    SUPPORTED_OCIGENAI_EMBEDDING_MODELS, \
    SUPPORTED_OCIGENAI_CHAT_MODELS, \
    EMBED_TRUNCATE

import numpy as np
import requests
import tiktoken
from fastapi import HTTPException

from api.models.base import BaseChatModel, BaseEmbeddingsModel
from api.schema import (
    # Chat
    ChatResponse,
    ChatRequest,
    Choice,
    ChatResponseMessage,
    Usage,
    ChatStreamResponse,
    ImageContent,
    TextContent,
    ToolCall,
    ChoiceDelta,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Function,
    ResponseFunction,
    # Embeddings
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    Embedding,
)


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **CLIENT_KWARGS
            )

ENCODER = tiktoken.get_encoding("cl100k_base")


class OCIGenAIModel(BaseChatModel):
    # https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm
    # https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm
    _supported_models = {}

    for model in SUPPORTED_OCIGENAI_CHAT_MODELS:
        _supported_models[model] = {
            "system": True,
            "multimodal": False,
            "tool_call": False,
            "stream_tool_call": False,
        }

    def list_models(self) -> list[str]:
        return list(self._supported_models.keys())

    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # check if model is supported
        if chat_request.model not in self._supported_models.keys():
            error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"

        # check if tool call is supported
        elif chat_request.tools and not self._is_tool_call_supported(chat_request.model, stream=chat_request.stream):
            tool_call_info = "Tool call with streaming" if chat_request.stream else "Tool call"
            error = f"{tool_call_info} is currently not supported by {chat_request.model}"

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    def _invoke_genai(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke OCI GenAI models"""
        if DEBUG:
            logger.info("Raw request: " + chat_request.model_dump_json())

        # convert OpenAI chat request to OCI Generative AI SDK request
        args = self._parse_request(chat_request)        
        if DEBUG:
            logger.info("OCI Generative AI request: " + json.dumps(args,ensure_ascii=False))
        try:             
            region = SUPPORTED_OCIGENAI_CHAT_MODELS[chat_request.model]["region"]
            # generative_ai_inference_client.base_client.config["region"] = region
            generative_ai_inference_client.base_client._endpoint = INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", region)
            response = generative_ai_inference_client.chat(args)
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        return response
    
    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        # message_id = self.generate_message_id()
        response = self._invoke_genai(chat_request)
        message_id = response.request_id

        if response.data.model_id.startswith("cohere"):
            texts = [{"text":response.data.chat_response.text}]
            finish_reason=response.data.chat_response.finish_reason
        elif response.data.model_id.startswith("meta"):
            texts = [{"text":c.message.content[0].text} for c in response.data.chat_response.choices]
            finish_reason=response.data.chat_response.choices[-1].finish_reason

        chat_response = self._create_response(
            model=response.data.model_id,
            message_id=message_id,
            content=texts,
            finish_reason=finish_reason,
            input_tokens=0,
            output_tokens=0,
        )
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())
        return chat_response

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        response = self._invoke_genai(chat_request, stream=True)
        # message_id = self.generate_message_id()
        message_id = response.request_id

        events = response.data.events()
        for stream in events:
            chunk = json.loads(stream.data)
            stream_response = self._create_response_stream(
                model_id=chat_request.model, message_id=message_id, chunk=chunk
            )
            if not stream_response:
                continue
            if DEBUG:
                logger.info("Proxy response :" + stream_response.model_dump_json())
            if stream_response.choices:
                yield self.stream_response_to_bytes(stream_response)
            elif (
                    chat_request.stream_options
                    and chat_request.stream_options.include_usage
            ):
                # An empty choices for Usage as per OpenAI doc below:
                # if you set stream_options: {"include_usage": true}.
                # an additional chunk will be streamed before the data: [DONE] message.
                # The usage field on this chunk shows the token usage statistics for the entire request,
                # and the choices field will always be an empty array.
                # All other chunks will also include a usage field, but with a null value.
                yield self.stream_response_to_bytes(stream_response)

        # return an [DONE] message at the end.
        yield self.stream_response_to_bytes()

    def _parse_system_prompts(self, chat_request: ChatRequest) -> list[dict[str, str]]:
        """Create system prompts.
        Note that not all models support system prompts.

        example output: [{"text" : system_prompt}]

        See example:
        https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
        """

        system_prompts = []
        for message in chat_request.messages:
            if message.role != "system":
                # ignore system messages here
                continue
            assert isinstance(message.content, str)
            system_prompts.append(message.content)            

        return system_prompts

    def _parse_messages(self, chat_request: ChatRequest) -> list[dict]:
        """
        Converse API only support user and assistant messages.

        example output: [{
            "role": "user",
            "content": [{"text": input_text}]
        }]

        See example:
        https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
        """
        messages = []
        for message in chat_request.messages:
            if isinstance(message, UserMessage):
                messages.append(
                    {
                        "role": message.role,
                        "content": self._parse_content_parts(
                            message, chat_request.model
                        ),
                    }
                )
            elif isinstance(message, AssistantMessage):
                if message.content:
                    # Text message
                    messages.append(
                        {"role": message.role, "content": [{"text": message.content}]}
                    )
                else:
                    # Tool use message
                    tool_input = json.loads(message.tool_calls[0].function.arguments)
                    messages.append(
                        {
                            "role": message.role,
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": message.tool_calls[0].id,
                                        "name": message.tool_calls[0].function.name,
                                        "input": tool_input
                                    }
                                }
                            ],
                        }
                    )
            elif isinstance(message, ToolMessage):
                
                # Add toolResult to content
                # https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": message.tool_call_id,
                                    "content": [{"text": message.content}],
                                }
                            }
                        ],
                    }
                )

            else:
                # ignore others, such as system messages
                continue
        return messages

    def _parse_request(self, chat_request: ChatRequest) -> dict:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
        """

        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)

        # Base inference parameters.        
        inference_config = {
            "maxTokens": chat_request.max_tokens,
            "isStream": chat_request.stream,                                    
            "frequencyPenalty": chat_request.frequency_penalty,
            "presencePenalty": chat_request.presence_penalty,
            "temperature": chat_request.temperature,
            "topP": chat_request.top_p,
            # "topK": chat_request.top_k
            }    

        model_name = chat_request.model        
        provider = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["provider"]
        compartment_id = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["compartment_id"]        

        if provider == "cohere":
            chatHistory = []
            for message in messages[:-1]:
                # print("="*22)
                # print(message)
                text = message["content"][0]["text"]
                #text = text.encode("unicode_escape").decode("utf-8")
                if message["role"] == "user":
                    chatHistory.append({"role": "USER", "message": text})
                elif message["role"] == "assistant":
                    chatHistory.append({"role": "CHATBOT", "message": text})
            text = messages[-1]["content"][0]["text"]
            #text = text.encode("unicode_escape").decode("utf-8")
            chatRequest =  {"apiFormat": "COHERE",
                            "preambleOverride":' '.join(system_prompts),
                            "message": text,
                            "chatHistory": chatHistory,
                            }
            chatRequest.update(inference_config)

        elif provider == "meta":
            meta_messages = []
            for message in messages:                
                text = message["content"][0]["text"]
                #text = text.encode("unicode_escape").decode("utf-8")                
                meta_messages.append({"role": message["role"].upper(), 
                                      "content": [{"type": "TEXT","text": text}]
                                     })
            chatRequest =   {                                    
                            "apiFormat": "GENERIC",
                            "messages": meta_messages,
                            "numGenerations": chat_request.n,
                            "topK":-1
                            }
            chatRequest.update(inference_config)        
        

        if provider == "dedicated":
            endpoint = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["endpoint"]
            servingMode ={"endpointId": endpoint, "servingType": "DEDICATED"}
        else:
            model_id = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["model_id"]
            servingMode ={"modelId": model_id, "servingType": "ON_DEMAND"}
                      
        chat_detail = {
            "compartmentId": compartment_id,
            "servingMode": servingMode,
            "chatRequest": chatRequest
            }


    #     # add tool config
    #     if chat_request.tools:
    #         chat_detail["toolCalls"] = []
            
            
    #         for t in chat_request.tools:
    #             t.function

    # #       {
    # #             "name": func.name,
    # #             "description": func.description,
    # #             "parameter_definitions": {
    # #                 "type":
    # #                 "description":
    # #                 "is_required":
    # #                 "json": func.parameters,
    # #             }
    # #         }

    #         if chat_request.tool_choice:
    #             if isinstance(chat_request.tool_choice, str):
    #                 # auto (default) is mapped to {"auto" : {}}
    #                 # required is mapped to {"any" : {}}
    #                 if chat_request.tool_choice == "required":
    #                     args["toolConfig"]["toolChoice"] = {"any": {}}
    #                 else:
    #                     args["toolConfig"]["toolChoice"] = {"auto": {}}
    #             else:
    #                 # Specific tool to use
    #                 assert "function" in chat_request.tool_choice
    #                 args["toolConfig"]["toolChoice"] = {
    #                     "tool": {"name": chat_request.tool_choice["function"].get("name", "")}}
        return chat_detail

    def _create_response(
            self,
            model: str,
            message_id: str,
            content: list[dict] = None,
            finish_reason: str | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ) -> ChatResponse:

        message = ChatResponseMessage(
            role="assistant",
        )
        if finish_reason == "tool_use":
            # https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/datatypes/CohereChatRequest
            tool_calls = []
            for part in content:
                if "toolUse" in part:
                    tool = part["toolUse"]
                    tool_calls.append(
                        ToolCall(
                            id=tool["toolUseId"],
                            type="function",
                            function=ResponseFunction(
                                name=tool["name"],
                                arguments=json.dumps(tool["input"]),
                            ),
                        )
                    )
            message.tool_calls = tool_calls
            message.content = None
        else:
            message.content = content[0]["text"]

        response = ChatResponse(
            id=message_id,
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason=self._convert_finish_reason(finish_reason),
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        response.system_fingerprint = "fp"
        response.object = "chat.completion"
        response.created = int(time.time())
        return response

    def _create_response_stream(
            self, model_id: str, message_id: str, chunk: dict
    ) -> ChatStreamResponse | None:
        """Parsing the OCI GenAI stream response chunk.

        Ref: https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
        """
        if DEBUG:
            logger.info("OCI GenAI response chunk: " + str(chunk))

        finish_reason = None
        message = None
        usage = None
        if "finishReason" not in chunk:
            if model_id.startswith("cohere"):
                text = chunk["text"]
            elif model_id.startswith("meta"):
                text = chunk["message"]["content"][0]["text"]
            message = ChatResponseMessage(
                role="assistant",
                content=text,
            )

        if "contentBlockStart" in chunk:
            # tool call start
            delta = chunk["contentBlockStart"]["start"]
            if "toolUse" in delta:
                # first index is content
                index = chunk["contentBlockStart"]["contentBlockIndex"] - 1
                message = ChatResponseMessage(
                    tool_calls=[
                        ToolCall(
                            index=index,
                            type="function",
                            id=delta["toolUse"]["toolUseId"],
                            function=ResponseFunction(
                                name=delta["toolUse"]["name"],
                                arguments="",
                            ),
                        )
                    ]
                )

        if "finishReason" in chunk:
            message = ChatResponseMessage()
            finish_reason = chunk["finishReason"]

        if "metadata" in chunk:
            # usage information in metadata.
            metadata = chunk["metadata"]
            if "usage" in metadata:
                # token usage
                return ChatStreamResponse(
                    id=message_id,
                    model=model_id,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=metadata["usage"]["inputTokens"],
                        completion_tokens=metadata["usage"]["outputTokens"],
                        total_tokens=metadata["usage"]["totalTokens"],
                    ),
                )
        if message:
            return ChatStreamResponse(
                id=message_id,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta=message,
                        logprobs=None,
                        finish_reason=self._convert_finish_reason(finish_reason),
                    )
                ],
                usage=usage,
            )

        return None



    def _parse_content_parts(
            self,
            message: UserMessage,
            model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self._is_multimodal_supported(model_id):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(
                    {
                        "image": {
                            "format": content_type[6:],  # image/
                            "source": {"bytes": image_data},
                        },
                    }
                )
            else:
                # Ignore..
                continue
        return content_parts

    def _is_tool_call_supported(self, model_id: str, stream: bool = False) -> bool:
        feature = self._supported_models.get(model_id)
        if not feature:
            return False
        return feature["stream_tool_call"] if stream else feature["tool_call"]

    def _is_multimodal_supported(self, model_id: str) -> bool:
        feature = self._supported_models.get(model_id)
        if not feature:
            return False
        return feature["multimodal"]

    def _is_system_prompt_supported(self, model_id: str) -> bool:
        feature = self._supported_models.get(model_id)
        if not feature:
            return False
        return feature["system"]

    # def _convert_tool_spec(self, func: Function) -> dict:


    #     return {
    #             "name": func.name,
    #             "description": func.description,
    #             "parameter_definitions": {
    #                 "type":
    #                 "description":
    #                 "is_required":
    #                 "json": func.parameters,
    #             }
    #         }
       

    def _convert_finish_reason(self, finish_reason: str | None) -> str | None:
        """
        Below is a list of finish reason according to OpenAI doc:

        - stop: if the model hit a natural stop point or a provided stop sequence,
        - length: if the maximum number of tokens specified in the request was reached,
        - content_filter: if content was omitted due to a flag from our content filters,
        - tool_calls: if the model called a tool
        """
        if finish_reason:            
            finish_reason_mapping = {
                "tool_use": "tool_calls",
                "COMPLETE":"stop",
                "ERROR_TOXIC":"content_filter",
                "ERROR_LIMIT":"stop",
                "ERROR":"stop",
                "USER_CANCEL":"stop",
                "MAX_TOKENS":"length",
            }
            return finish_reason_mapping.get(finish_reason.lower(), finish_reason.lower())
        return None


class OCIGenAIEmbeddingsModel(BaseEmbeddingsModel, ABC):
    accept = "application/json"
    content_type = "application/json"
    

    def _invoke_model(self, args: dict, model_id: str):
        # body = json.dumps(args)
        compartment_id = SUPPORTED_OCIGENAI_EMBEDDING_MODELS[model_id]["compartment_id"]
        region = SUPPORTED_OCIGENAI_EMBEDDING_MODELS[model_id]["region"]
        generative_ai_inference_client.base_client._endpoint = INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", region)
        body = {
                "inputs": args["texts"],
                "servingMode": {"servingType": "ON_DEMAND","modelId": model_id},
                "truncate": args["truncate"],
                "compartmentId": compartment_id
                }
        if DEBUG:
            logger.info("Invoke OCI GenAI Model: " + model_id)
            logger.info("OCI GenAI request body: " + json.dumps(body))
        try:

            embed_text_response = generative_ai_inference_client.embed_text(body)
            return embed_text_response        

        except Exception as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))

    def _create_response(
            self,
            embeddings: list[float],
            model: str,
            input_tokens: int = 0,
            output_tokens: int = 0,
            encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingsResponse:
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))


        response = EmbeddingsResponse(
            data=data,
            model=model,
            usage=EmbeddingsUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        if DEBUG:
            logger.info("Proxy response :" + response.model_dump_json()[:100])
        return response


class CohereEmbeddingsModel(OCIGenAIEmbeddingsModel):

    def _parse_args(self, embeddings_request: EmbeddingsRequest) -> dict:
        texts = []
        if isinstance(embeddings_request.input, str):
            texts = [embeddings_request.input]
        elif isinstance(embeddings_request.input, list):
            texts = embeddings_request.input
        elif isinstance(embeddings_request.input, Iterable):
            # For encoded input
            # The workaround is to use tiktoken to decode to get the original text.
            encodings = []
            for inner in embeddings_request.input:
                if isinstance(inner, int):
                    # Iterable[int]
                    encodings.append(inner)
                else:
                    # Iterable[Iterable[int]]
                    text = ENCODER.decode(list(inner))
                    texts.append(text)
            if encodings:
                texts.append(ENCODER.decode(encodings))

        # Maximum of 2048 characters
        args = {
            "texts": texts,
            "input_type": "search_document",
            "truncate": EMBED_TRUNCATE,  # "NONE|START|END"
        }
        return args

    def embed(self, embeddings_request: EmbeddingsRequest) -> EmbeddingsResponse:
        response = self._invoke_model(
            args=self._parse_args(embeddings_request), model_id=embeddings_request.model
        )
        response_body = response.data
        if DEBUG:
            logger.info("OCI GenAI response body: " + str(response_body)[:50])

        return self._create_response(
            embeddings=response_body.embeddings,
            model=response_body.model_id,
            encoding_format=embeddings_request.encoding_format,
        )


def get_embeddings_model(model_id: str) -> OCIGenAIEmbeddingsModel:
    model_name = SUPPORTED_OCIGENAI_EMBEDDING_MODELS.get(model_id, "")
    if DEBUG:
        logger.info("model name is " + model_name.get("model_id", model_name))
    if model_name:
        return CohereEmbeddingsModel()
    else:
        logger.error("Unsupported model id " + model_id)
        raise HTTPException(
            status_code=400,
            detail="Unsupported embedding model id " + model_id,
        )
