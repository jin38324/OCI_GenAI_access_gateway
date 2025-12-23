import json
import logging
import copy

from typing import AsyncIterable
from fastapi import HTTPException
import requests

from api.setting import (
    DEBUG, 
    CLIENT_KWARGS, 
    INFERENCE_ENDPOINT_TEMPLATE,
    INFERENCE_ENDPOINT_TEMPLATE_OPENAI,
    SUPPORTED_OCIGENAI_CHAT_MODELS,
    OCI_REGION,
    OCI_COMPARTMENT
)

from api.models.base import BaseChatModel
from api.models.utils import logger
from api.schema import ChatRequest

from api.models.adapter.request_adapter import ChatRequestAdapter
from api.models.adapter.response_adapter import ResponseAdapter

from openai.types.chat.chat_completion import ChatCompletion

from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai import GenerativeAiClient
from oci.generative_ai_inference import models as oci_models


class OCIGenAIModel(BaseChatModel):
    # https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm
    # https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm

    def __init__(self):
        self.provider = ""
        self.generative_ai_inference_client = GenerativeAiInferenceClient(**CLIENT_KWARGS)
        self.init_models()

    def init_models(self):
        def capability_filter(capabilities):
            if "CHAT" in capabilities \
                or "TEXT_GENERATION" in capabilities \
                or "TEXT_TO_TEXT" in capabilities:
                return True
            else:
                return False

        if not SUPPORTED_OCIGENAI_CHAT_MODELS:
            list_models_response, list_imported_models_response, list_endpoints_response = self.list_models(retrive=True)
            for model in list_models_response:
                if capability_filter(model.capabilities):
                    provider = model.display_name.split(".")[0] if "." in model.display_name else "UNKNOWN"
                    SUPPORTED_OCIGENAI_CHAT_MODELS[model.display_name] = {
                        "type":"ondemand",
                        "model_id":model.display_name,
                        "provider":provider,
                        "region": OCI_REGION,
                        "compartment_id": OCI_COMPARTMENT,
                    }
            if list_imported_models_response:
                for model in list_imported_models_response:
                    if capability_filter(model.capabilities):
                        for item in list_endpoints_response:
                            if item.model_id==model.id:
                                SUPPORTED_OCIGENAI_CHAT_MODELS[model.display_name] = {                                    
                                    "type": "dedicated",
                                    "model_id": model.display_name,
                                    "provider": "generic",
                                    "endpoint": item.id,
                                    "region": OCI_REGION,
                                    "compartment_id": OCI_COMPARTMENT,
                                }

            logger.info(f"Successfully get {len(SUPPORTED_OCIGENAI_CHAT_MODELS)} models")

    def list_models(self, retrive: bool = False) -> list:
        try:
            if retrive:
                CLIENT_KWARGS.update({'service_endpoint':   f"https://generativeai.{CLIENT_KWARGS['region']}.oci.oraclecloud.com" })
                generative_ai_client = GenerativeAiClient(**CLIENT_KWARGS)
                # get ondemand models
                list_models_response = generative_ai_client.list_models(
                    compartment_id=OCI_COMPARTMENT,
                    lifecycle_state = "ACTIVE"
                )
                # get imported models
                list_imported_models_response = generative_ai_client.list_imported_models(
                    compartment_id=OCI_COMPARTMENT,
                    lifecycle_state = "ACTIVE"
                )
                # get imported model endpoints    
                list_endpoints_response = generative_ai_client.list_endpoints(
                    compartment_id=OCI_COMPARTMENT,    
                    lifecycle_state = "ACTIVE"
                )

                return list_models_response.data.items, list_imported_models_response.data.items, list_endpoints_response.data.items
            else:
                return list(SUPPORTED_OCIGENAI_CHAT_MODELS.keys())
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))


    def validate(self, chat_request: ChatRequest):
        """Perform basic validation on requests"""
        error = ""
        # check if model is supported
        if chat_request.model not in SUPPORTED_OCIGENAI_CHAT_MODELS.keys():
            error = f"Unsupported model {chat_request.model}, please use models API to get a list of supported models"

        if error:
            raise HTTPException(
                status_code=400,
                detail=error,
            )

    def _log_chat(self,chat_request):
        def modify_msg(messages):            
            for message in messages:                
                try:
                    if isinstance(message, dict):
                        if isinstance(message["content"], list):
                            for c in message["content"]:
                                if c["type"] == "image_url":
                                    c["image_url"]["url"] = c["image_url"]["url"][ :50] + "..."
                    
                    else:
                        if not isinstance(message.content, str):
                            for c in message.content:
                                if c.type == "IMAGE":
                                    c.image_url["url"] = c.image_url["url"][ :50] + "..."
                except Exception as e:
                    logging.info("Warning:"+str(e))
                    print(message)  
            return messages
        try:
            temp_chat_request = copy.deepcopy(chat_request)
            if isinstance(temp_chat_request, ChatRequest):
                # modify openai message
                temp_chat_request.messages = modify_msg(temp_chat_request.messages)
                return temp_chat_request.model_dump_json(indent=2)
            elif isinstance(temp_chat_request.chat_request, oci_models.GenericChatRequest):
                # modify oci generic message
                temp_chat_request.chat_request.messages = modify_msg(temp_chat_request.chat_request.messages)
                return str(temp_chat_request)
            else:
                return str(chat_request)
        except Exception as e:
            logging.info("Failed to convert log chat request:"+str(e))
            return str(chat_request)
        

    def _invoke_genai(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke OCI GenAI models"""
        if DEBUG:
            logger.info("Raw request:\n" + self._log_chat(chat_request))

        
        model_name = chat_request.model
        model_info = SUPPORTED_OCIGENAI_CHAT_MODELS[model_name]        
        region = model_info["region"]
        self.provider = model_info["provider"]

        # use openai compatitble API
        if self.provider in ["meta","xai"]:
            response = requests.post(
                url = INFERENCE_ENDPOINT_TEMPLATE_OPENAI.replace("{region}", region),
                auth=CLIENT_KWARGS["signer"],
                data= json.dumps(chat_request.model_dump()),
                headers={
                    "content-type": "application/json",
                    "CompartmentId": OCI_COMPARTMENT
                }
            )
            return response

        # use generic API
        else:
            # convert OpenAI chat request to OCI Generative AI SDK request
            chat_detail = ChatRequestAdapter(model_info).to_oci(chat_request)
            if DEBUG:
                logger.info("OCI Generative AI request:\n" + self._log_chat(chat_detail))                
            
            self.generative_ai_inference_client.base_client._endpoint = INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", region)
            response = self.generative_ai_inference_client.chat(chat_detail)
            if DEBUG and not chat_detail.chat_request.is_stream:
                info = str(response.data)
                logger.info("OCI Generative AI response:\n" + info) 
            return response

    def chat(self, chat_request: ChatRequest) -> ChatCompletion:
        """Default implementation for Chat API."""
        response = self._invoke_genai(chat_request)

        if self.provider in ["meta","xai"]:
            chat_response = json.loads(response.text)
            info = json.dumps(chat_response, indent=2, ensure_ascii=False)
        else:            
            # message_id = self.generate_message_id()
            message_id = response.request_id
            model_id = chat_request.model
            chat_response = ResponseAdapter(self.provider).to_openai(
                model=model_id,
                message_id=message_id,
                response=response.data.chat_response
                )            
            info = chat_response.model_dump_json(indent=2)
        if DEBUG:
            logger.info("Proxy response :" +info)
        return chat_response

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        response = self._invoke_genai(chat_request)
        if self.provider in ["meta","xai"]:
            for chunk in response:
                if DEBUG:
                    logger.info("Proxy response :" + chunk.decode('utf-8-sig', errors="replace"))
                yield chunk
        else:
            message_id = response.request_id
            model_id = SUPPORTED_OCIGENAI_CHAT_MODELS[chat_request.model]["model_id"]
            
            for stream in response.data.events():
                chunk = json.loads(stream.data)
                if DEBUG:
                    logger.info("OCI response :" + str(chunk))
                stream_response = ResponseAdapter(self.provider).to_openai_chunk(
                    message_id=message_id,
                    model=model_id,
                    chunk=chunk
                    )
                if DEBUG:
                    logger.info("Proxy response :" + stream_response.model_dump_json())

                yield self.stream_response_to_bytes(stream_response)

            # return an [DONE] message at the end.
            yield self.stream_response_to_bytes()



