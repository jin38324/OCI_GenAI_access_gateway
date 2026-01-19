import json
import logging
import copy

from typing import AsyncIterable
from fastapi import HTTPException
from fastapi.openapi.models import RequestBody
import requests
from requests.utils import dict_to_sequence

from api.setting import (
    DEBUG, 
    CLIENT_KWARGS, 
    INFERENCE_ENDPOINT_TEMPLATE,
    INFERENCE_ENDPOINT_TEMPLATE_OPENAI,
    INFERENCE_ENDPOINT_TEMPLATE_RESPONSES,
    SUPPORTED_OCIGENAI_CHAT_MODELS,
    OCI_REGION,
    OCI_COMPARTMENT
)

from api.models.base import BaseChatModel
from api.models.utils import logger,trim_image
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

    def __init__(self, model_id:str=None):
        self.model_id = model_id
        self.provider = self._get_provider(model_id)
        self.compatitble_providers = ["meta","xai","openai"]        
        self.init_models()
        self.region = self._get_region()

    def _get_provider(self, model_id):
        if model_id is None:
            return "UNKNOWN"
        elif "." in model_id:
            return model_id.split(".")[0]
        else:
            return "UNKNOWN"

    def _get_region(self):
        if self.model_id is None:
            return OCI_REGION
        else:
            model_info = SUPPORTED_OCIGENAI_CHAT_MODELS[self.model_id]        
            region = model_info["region"]
            return region

    def _log_chat(self,header,content):
        if DEBUG:
            logger.info(header + ":\n" + trim_image(content))

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
                    SUPPORTED_OCIGENAI_CHAT_MODELS[model.display_name] = {
                        "type":"ondemand",
                        "model_id":model.display_name,
                        "provider": self._get_provider(model.display_name),
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


    def validate(self):
        """Perform basic validation on requests"""
        # check if model is supported
        if self.model_id not in SUPPORTED_OCIGENAI_CHAT_MODELS.keys():        
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model {self.model_id}, please use models API to get a list of supported models",
            )
      

    def _invoke_genai(self, chat_request: ChatRequest, stream=False):
        """Common logic for invoke OCI GenAI models"""
        

        # use openai compatitble API
        if self.provider in self.compatitble_providers:
            self._log_chat("OCI Generative AI request",chat_request.model_dump())
            http_client_headers = self._build_headers(compartment_id = OCI_COMPARTMENT)
            response = requests.post(
                url = INFERENCE_ENDPOINT_TEMPLATE_OPENAI.replace("{region}", self.region),
                auth=CLIENT_KWARGS["signer"],
                data= json.dumps(chat_request.model_dump()),
                headers=http_client_headers
            )
            return response

        # use generic API
        else:
            # convert OpenAI chat request to OCI Generative AI SDK request
            model_info = SUPPORTED_OCIGENAI_CHAT_MODELS[self.model_id]
            chat_detail = ChatRequestAdapter(model_info).to_oci(chat_request)
            
            self._log_chat("OCI Generative AI request",chat_detail)             
            
            generative_ai_inference_client = GenerativeAiInferenceClient(**CLIENT_KWARGS)
            generative_ai_inference_client.base_client._endpoint = INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", self.region)
            response = generative_ai_inference_client.chat(chat_detail)
            if not chat_detail.chat_request.is_stream:
                self._log_chat("OCI Response",str(response.data))
            return response

    def chat(self, chat_request: ChatRequest) -> ChatCompletion:
        """Default implementation for Chat API."""
        response = self._invoke_genai(chat_request)

        if self.provider in self.compatitble_providers:
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
        self._log_chat("OCI Response",info)
        return chat_response

    def chat_stream(self, chat_request: ChatRequest) -> AsyncIterable[bytes]:
        """Default implementation for Chat Stream API"""
        response = self._invoke_genai(chat_request)
        if self.provider in self.compatitble_providers:
            for chunk in response:
                if DEBUG:
                    logger.info("OCI Response :" + chunk.decode('utf-8-sig', errors="replace"))
                yield chunk
        else:
            message_id = response.request_id
            model_id = SUPPORTED_OCIGENAI_CHAT_MODELS[chat_request.model]["model_id"]
            
            for stream in response.data.events():
                chunk = json.loads(stream.data)
                if DEBUG:
                    logger.info("OCI Response :" + str(chunk))
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


    def _invoke_response(self, chat_request: dict, stream=False):
        """Common logic for invoke OCI GenAI models"""
        chat_request.setdefault("store", False)

        self._log_chat("Raw request",chat_request)

        # use openai compatitble API
        if self.provider in self.compatitble_providers:
            http_client_headers = self._build_headers(OCI_COMPARTMENT, chat_request.get("conversation"))

            response = requests.post(
                url = INFERENCE_ENDPOINT_TEMPLATE_RESPONSES.replace("{region}", self.region),
                auth = CLIENT_KWARGS["signer"],
                data = json.dumps(chat_request),
                headers = http_client_headers
            )
            return response
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented responses API.")

    def responses(self, chat_request: dict) -> dict:
        response = self._invoke_response(chat_request)
        chat_response = json.loads(response.text)
        # Fix annotation bug
        if "output" in chat_response:
            for i in chat_response["output"]:
                if "content" in i:
                    for j in i["content"]:
                        j["annotations"] = []

        self._log_chat("OCI Response",chat_response)
        return chat_response

    def responses_stream(self, chat_request: dict) -> AsyncIterable[bytes]:
        response = self._invoke_response(chat_request)
        for chunk in response:
            if DEBUG:
                logger.info("OCI Response :" + chunk.decode('utf-8-sig', errors="replace"))
            yield chunk


    def _build_headers(self,compartment_id: str = None, conversation_store_id: str = None):
        http_client_headers = (
            {
                # "CompartmentId": compartment_id,  # for backward compatibility
                "opc-compartment-id": compartment_id,
            }
            if compartment_id
            else {}
        )
        if conversation_store_id:
            http_client_headers["opc-conversation-store-id"] = conversation_store_id
        return http_client_headers