import base64
import json
import logging
import re
import copy
import time
from abc import ABC
from typing import AsyncIterable, Iterable, Literal

import oci
from oci.generative_ai_inference import models as oci_models
from api.setting import DEBUG
from api.setting import CLIENT_KWARGS, \
    INFERENCE_ENDPOINT_TEMPLATE, \
    SUPPORTED_OCIGENAI_EMBEDDING_MODELS, \
    SUPPORTED_OCIGENAI_CHAT_MODELS

import numpy as np
import requests
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
    Convertor
)
from config import EMBED_TRUNCATE
from .call_api import patched_call_api
from oci.base_client import BaseClient 
BaseClient.call_api = patched_call_api

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    **CLIENT_KWARGS
)


class OCIGenAIModel(BaseChatModel):
    # https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm
    # https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm

    _supported_models = {}

    for model in SUPPORTED_OCIGENAI_CHAT_MODELS:
        model_setting = SUPPORTED_OCIGENAI_CHAT_MODELS[model]
        _supported_models[model] = {
            "system": model_setting.get('system', True),
            "multimodal": model_setting.get('multimodal', False),
            "tool_call": model_setting.get('tool_call', False),
            "stream_tool_call": model_setting.get('stream_tool_call', False),
        }

    def list_models(self) -> list[str]:
        try:
            generative_ai_client = oci.generative_ai.GenerativeAiClient(
                **CLIENT_KWARGS
            )
            first_key = next(iter(SUPPORTED_OCIGENAI_CHAT_MODELS))
            compartment_id = SUPPORTED_OCIGENAI_CHAT_MODELS[first_key]["compartment_id"]
            list_models_response = generative_ai_client.list_models(
                    compartment_id=compartment_id,
                    capability=["TEXT_GENERATION"]
                    )
            # valid_models = [model.display_name for model in list_models_response.data.items]
            logger.info("Successfully validated models")
            return list(self._supported_models.keys())
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))


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
            temp_chat_request = copy.deepcopy(chat_request)
            for message in temp_chat_request.messages:
                try:
                    for c in message.content:
                        if c.type == "image_url":
                            c.image_url.url = c.image_url.url[:50] + "..."
                except:
                    pass
            logger.info("Raw request:\n" + temp_chat_request.model_dump_json())

        # convert OpenAI chat request to OCI Generative AI SDK request
        chat_detail = self._parse_request(chat_request)
        if DEBUG:
            temp_chat_detail = copy.deepcopy(chat_detail)  
            try:                          
                for message in temp_chat_detail.chat_request.messages:
                    try:
                        for c in message.content:                    
                            if c.type == "IMAGE":
                                c.image_url["url"] = c.image_url["url"][:50] + "..."
                    except:
                        pass
            except Exception as e:
                    logging.info("Warning:"+str(e))
            logger.info("OCI Generative AI request:\n" + json.dumps(json.loads(str(temp_chat_detail)), ensure_ascii=False))
        try:
            region = SUPPORTED_OCIGENAI_CHAT_MODELS[chat_request.model]["region"]
            # generative_ai_inference_client.base_client.config["region"] = region
            generative_ai_inference_client.base_client._endpoint = INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", region)
            # response = generative_ai_inference_client.chat(chat_detail)
            
            body = generative_ai_inference_client.base_client.sanitize_for_serialization(chat_detail) 

            if "isStream" in body["chatRequest"]:
                if body["chatRequest"]["isStream"]:
                    body["chatRequest"]["streamOptions"] = {"isIncludeUsage": True}
            body = json.dumps(body)

            #response = client.chat(chat_detail)
            response = generative_ai_inference_client.base_client.call_api(
                resource_path="/actions/chat",
                method="POST",
                operation_name="chat",
                header_params={
                    "accept": "application/json, text/event-stream",
                    "content-type": "application/json"
                },
                body=body,
                #response_type="ChatResult"
                )

            if DEBUG and not chat_detail.chat_request.is_stream:
                content = json.dumps(json.loads(response.data.content), ensure_ascii=False)
                logger.info("OCI Generative AI response:\n" + content)
        #except oci.exceptions.ServiceError as e:
        #    logger.error(f"[_invoke_genai] OCI ServiceError: Status Code: {e.status}. Message: {e.message}")
        #    raise HTTPException(status_code=e.status, detail=e.message)        
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        return response

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Default implementation for Chat API."""

        # message_id = self.generate_message_id()
        response = self._invoke_genai(chat_request)
        message_id = response.request_id
        model_id = chat_request.model
        data = json.loads(response.data.content)
        prompt_tokens = data["chatResponse"]["usage"]["promptTokens"]
        total_tokens = data["chatResponse"]["usage"]["totalTokens"]
        completion_tokens = total_tokens - prompt_tokens
        try:
            chat_response = self._create_response(
                model=model_id,
                message_id=message_id,
                chat_response=data["chatResponse"],
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail="ERROR _create_response: " + str(e))        
        if DEBUG:
            logger.info("Proxy response :" + chat_response.model_dump_json())
        return chat_response

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        # print("="*20,str(chat_request))
        response = self._invoke_genai(chat_request)
        if not response.data:
            raise HTTPException(status_code=500, detail="OCI AI API returned empty response")

        # message_id = self.generate_message_id()
        message_id = response.request_id
        model_id = SUPPORTED_OCIGENAI_CHAT_MODELS[chat_request.model]["model_id"]        
        events = response.data.events()
        for stream in events:
            chunk = json.loads(stream.data)
            stream_response = self._create_response_stream(
                model_id=model_id, message_id=message_id, chunk=chunk
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
                elif message.tool_calls:
                    # Tool use message
                    # formate https://platform.openai.com/docs/guides/function-calling?api-mode=chat#handling-function-calls                    
                    messages.append({"role": message.role,"tool_calls": message.tool_calls})
            elif isinstance(message, ToolMessage):
                # Add toolResult to content
                # https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content
                    }
                )

            else:
                # ignore others, such as system messages
                continue
        return messages

    def _parse_request(self, chat_request: ChatRequest) -> oci_models.ChatDetails:
        """Create default converse request body.

        Also perform validations to tool call etc.

        Ref: https://docs.oracle.com/en-us/iaas/api/#/EN/generative-ai-inference/20231130/ChatResult/Chat
        """

        messages = self._parse_messages(chat_request)
        system_prompts = self._parse_system_prompts(chat_request)
        

        # Base inference parameters.        
        inference_config = {
            "max_tokens": chat_request.max_tokens,
            "is_stream": chat_request.stream,
            "frequency_penalty": chat_request.frequency_penalty,
            "presence_penalty": chat_request.presence_penalty,
            "temperature": chat_request.temperature,
            "top_p": chat_request.top_p
            }

        model_name = chat_request.model
        type = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["type"]
        provider = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["provider"]
        compartment_id = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["compartment_id"]

        if type == "dedicated":
            endpoint = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["endpoint"]
            servingMode = oci_models.DedicatedServingMode(
                serving_type = "DEDICATED",
                endpoint_id = endpoint
                )
        elif type == "ondemand":
            model_id = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]["model_id"]
            servingMode = oci_models.OnDemandServingMode(
                serving_type = "ON_DEMAND",
                model_id = model_id
                )
        chat_detail = oci_models.ChatDetails(
            compartment_id = compartment_id,
            serving_mode = servingMode,
            # chat_request = chatRequest
            )
        
        if provider == "cohere":
            cohere_chatRequest = oci_models.CohereChatRequest(**inference_config)
            if system_prompts:
                cohere_chatRequest.preamble_override = ' '.join(system_prompts)
            
            # add tools
            if chat_request.tools:
                cohere_tools = Convertor.convert_tools_openai_to_cohere(chat_request.tools)
                cohere_chatRequest.tools = cohere_tools  
            
            chatHistory = []
            for i,message in enumerate(messages):
                # process chat history
                if i < len(messages)-1:                
                    # print("="*22,'\n',message)
                    # text = text.encode("unicode_escape").decode("utf-8")
                    try:
                        text = message["content"][0]["text"]
                    except:
                        text = ""               
                    if message["role"] == "user":
                        message_line = oci_models.CohereUserMessage(
                            role = "USER",
                            message = text
                            )             
                    elif message["role"] == "assistant":
                        if "tool_calls" in message:
                            if not message["tool_calls"]:
                                message_line = oci_models.CohereChatBotMessage(
                                    role = "CHATBOT",
                                    message = text
                                    )
                            else:
                                message_line = oci_models.CohereChatBotMessage(
                                    role = "CHATBOT",
                                    message = text,
                                    tool_calls = Convertor.convert_tool_calls_openai_to_cohere(message["tool_calls"])
                                    ) 
                        else:
                            message_line = oci_models.CohereChatBotMessage(
                                    role = "CHATBOT",
                                    message = text
                                    )                 

                    elif message["role"] == "tool":
                        cohere_tool_results = []
                        cohere_tool_result = Convertor.convert_tool_result_openai_to_cohere(message)
                        cohere_tool_results.append(cohere_tool_result)
                        message_line = oci_models.CohereToolMessage(
                            role = "TOOL",
                            tool_results = cohere_tool_results
                            )
                        
                    chatHistory.append(message_line)
                # process the last message    
                elif i == len(messages)-1:
                    if message["role"] in ("user","assistant","system"):
                        cohere_chatRequest.message = message["content"][0]["text"]
                        # text = text.encode("unicode_escape").decode("utf-8")
                    # input tool result
                    elif message["role"] == "tool":
                        cohere_chatRequest.message = ""
                        cohere_tool_results = []
                        cohere_tool_result = Convertor.convert_tool_result_openai_to_cohere(message)
                        cohere_tool_results.append(cohere_tool_result)
                        cohere_chatRequest.tool_results = cohere_tool_results

                cohere_chatRequest.chat_history = chatHistory
            chat_detail.chat_request = cohere_chatRequest

        elif provider == "meta" or provider == "openai" or provider == "xai":
            generic_chatRequest = oci_models.GenericChatRequest(**inference_config)
            generic_chatRequest.numGenerations = chat_request.n
            generic_chatRequest.topK = -1
            
            # add tools
            if chat_request.tools:
                llama_tools = Convertor.convert_tools_openai_to_llama(chat_request.tools)
                generic_chatRequest.tools = llama_tools

            meta_messages = []

            if system_prompts:
                meta_message = oci_models.SystemMessage(
                    role = "SYSTEM",
                    content = [oci_models.TextContent(type = "TEXT",text = ' '.join(system_prompts))]
                )
                meta_messages.append(meta_message)

            for message in messages:
                message["role"] = message["role"].upper()                

                if message["role"] == "USER":
                    content = []
                    for c in message["content"]:
                        if c["type"] == "TEXT":
                            content.append(oci_models.TextContent(type = "TEXT",text = c["text"]))
                        elif c["type"] == "IMAGE":
                            content.append(oci_models.ImageContent(type = "IMAGE",image_url  = c["imageUrl"]))                        

                    meta_message = oci_models.UserMessage(
                        role = "USER",
                        content = content
                        )
                
                elif message["role"] == "TOOL":
                    meta_message = Convertor.convert_tool_result_openai_to_llama(message)

                elif message["role"] == "ASSISTANT":
                    content = None
                    tool_calls = None
                    if "content" in message:
                        if message["content"]:
                            content = [oci_models.TextContent(type = "TEXT",text = c["text"]) for c in message["content"]]
                    
                    if "tool_calls" in message:
                        if message["tool_calls"]:
                            tool_calls = Convertor.convert_tool_calls_openai_to_llama(message["tool_calls"])

                    meta_message = oci_models.AssistantMessage(
                        role = "ASSISTANT",
                        content = content,
                        tool_calls = tool_calls
                        )
                else:
                    meta_message = message

                meta_messages.append(meta_message)
            generic_chatRequest.messages = meta_messages
            chat_detail.chat_request = generic_chatRequest
        # print(chat_detail)
        return chat_detail

    def _create_response(
            self,
            model: str,
            message_id: str,
            # content: list[dict] = None,
            chat_response = None,
            # finish_reason: str | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
    ) -> ChatResponse:
        message = ChatResponseMessage(role="assistant")
        if model.startswith("cohere"):
            finish_reason = chat_response["finishReason"]
            if "toolCalls" in chat_response:
                oepnai_tool_calls = Convertor.convert_tool_calls_to_openai(chat_response["toolCalls"],vendor="cohere")
                message.tool_calls = oepnai_tool_calls
                message.content = None
            else:
                message.content = chat_response["text"]
        else:            
            choice = chat_response["choices"][-1]
            finish_reason = choice["finishReason"]
            if "toolCalls" in choice["message"]:
                response_tool_calls = choice["message"]["toolCalls"]
            else:
                response_tool_calls = None
            if finish_reason == "tool_calls" or response_tool_calls:
                oepnai_tool_calls = Convertor.convert_tool_calls_to_openai(response_tool_calls)
                message.tool_calls = oepnai_tool_calls
                message.content = None
            else:
                message.content = choice["message"]["content"][0]["text"]

        response = ChatResponse(
            id = message_id,
            model = model,
            choices = [
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
        text = None
        openai_tool_calls = None
        vendor = model_id.split(".")[0].lower()
        if "finishReason" not in chunk:
            if vendor == "cohere":
                if "tooCalls" not in chunk:
                    text = chunk["text"]
                    message = ChatResponseMessage(
                        role="assistant",
                        content=text,
                        tool_calls=openai_tool_calls
                        )
                elif "toolCalls" in chunk:
                    # pass
                    openai_tool_calls = Convertor.convert_tool_calls_cohere_to_openai(chunk["toolCalls"])
                    message = ChatResponseMessage(
                        role="assistant",
                        tool_calls=openai_tool_calls
                        )
            else:
                if chunk.get("message", {}).get("content", [{}])[0].get("text"):
                    text = chunk["message"]["content"][0]["text"]
                if  chunk.get("message", {}).get("toolCalls"):
                    openai_tool_calls = Convertor.convert_tool_calls_to_openai(chunk["message"]["toolCalls"])
                message = ChatResponseMessage(
                    role="assistant",
                    content=text,
                    tool_calls=openai_tool_calls
                    )
        elif "finishReason" in chunk:
            message = ChatResponseMessage(role="assistant")
            finish_reason = chunk["finishReason"]
            if chunk.get("message", {}).get("toolCalls"):
                openai_tool_calls = Convertor.convert_tool_calls_cohere_to_openai(chunk["message"]["toolCalls"])
                message.tool_calls = openai_tool_calls
                message.content = ""

        if "usage" in chunk:
            # usage information in metadata.
            prompt_tokens = chunk["usage"]["promptTokens"]
            total_tokens = chunk["usage"]["totalTokens"]
            completion_tokens = total_tokens - prompt_tokens            
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        
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

        # return None

    def _parse_content_parts(
            self,
            message: UserMessage,
            model_id: str,
    ) -> list[dict]:
        if isinstance(message.content, str):
            return [
                {
                    "type": "TEXT",
                    "text": message.content,
                }
            ]
        content_parts = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_parts.append(
                    {
                        "type": "TEXT",
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImageContent):
                if not self._is_multimodal_supported(model_id):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multimodal message is currently not supported by {model_id}",
                    )
                # image_data, content_type = self._parse_image(part.image_url.url)
                content_parts.append(                    
                    {
                        "type": "IMAGE",
                        "imageUrl": {"url": f"{part.image_url.url}"},
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
                "COMPLETE": "stop",
                "ERROR_TOXIC": "content_filter",
                "ERROR_LIMIT": "stop",
                "ERROR": "stop",
                "USER_CANCEL": "stop",
                "MAX_TOKENS": "length",
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
        embed_text_details = oci.generative_ai_inference.models.EmbedTextDetails(
            compartment_id=compartment_id,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=model_id,
            ),
            # truncate = "NONE",
            inputs=args["texts"],
        )
        body = generative_ai_inference_client.base_client.sanitize_for_serialization(embed_text_details)
        body = json.dumps(body)
        if DEBUG: 
            logger.info("Invoke OCI GenAI Model: " + model_id)
            logger.info("OCI GenAI request body: " + str(body))
        try:
            embed_text_response = generative_ai_inference_client.base_client.call_api(
                resource_path="/actions/embedText",
                method="POST",
                operation_name="embedText",
                header_params={
                    "accept": "application/json, text/event-stream",
                    "content-type": "application/json"
                },
                body=body
                )
            return embed_text_response
        except Exception as e:
            logger.error("Validation Error: " + str(e))
            raise HTTPException(status_code=400, detail=str(e))

    def _create_response(
            self,
            embeddings: list[float],
            model: str,
            input_tokens: int = 0,
            total_tokens: int = 0,
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
                total_tokens=total_tokens,
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
        # elif isinstance(embeddings_request.input, Iterable):
        #     # For encoded input
        #     # The workaround is to use tiktoken to decode to get the original text.
        #     encodings = []
        #     for inner in embeddings_request.input:
        #         if isinstance(inner, int):
        #             # Iterable[int]
        #             encodings.append(inner)
        #         else:
        #             # Iterable[Iterable[int]]
        #             text = ENCODER.decode(list(inner))
        #             texts.append(text)
        #     if encodings:
        #         texts.append(ENCODER.decode(encodings))

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
        response_body = json.loads(response.data.text)
        input_tokens = response_body["usage"]["promptTokens"]
        total_tokens = response_body["usage"]["totalTokens"]
        if DEBUG:
            logger.info("OCI GenAI response body: " + str(response_body)[:200])

        return self._create_response(
            embeddings=response_body["embeddings"],
            model=response_body["modelId"],
            input_tokens=input_tokens,
            total_tokens=total_tokens,
            encoding_format=embeddings_request.encoding_format,
        )


def get_embeddings_model(model_id: str) -> OCIGenAIEmbeddingsModel:
    model_name = SUPPORTED_OCIGENAI_EMBEDDING_MODELS.get(model_id, "")
    if model_name:
        if DEBUG:
            logger.info("model name is " + model_name["name"])
        return CohereEmbeddingsModel()
    else:
        logger.error("Unsupported model id " + model_id)
        raise HTTPException(
            status_code=400,
            detail="Unsupported embedding model id " + model_id,
        )
