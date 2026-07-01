import logging
from abc import ABC
from datetime import datetime, timezone

from fastapi import HTTPException
from oci.generative_ai import GenerativeAiClient
from oci.generative_ai_inference import GenerativeAiInferenceClient

from api.models.adapter.rerank_adapter import RerankRequestAdapter
from api.models.utils import logger
from api.schema import RerankRequest, RerankResponse
from api.setting import (
    CLIENT_KWARGS,
    DEBUG,
    INFERENCE_ENDPOINT_TEMPLATE,
    OCI_COMPARTMENT,
    OCI_RERANK_DEFAULT_MODEL,
    OCI_REGION,
    SUPPORTED_OCIGENAI_RERANK_MODELS,
)


class OCIGenAIRerankModel(ABC):
    def __init__(self, model_id: str = None):
        self.model_id = model_id
        self.generative_ai_inference_client = GenerativeAiInferenceClient(**CLIENT_KWARGS)
        self.init_models()

    def init_models(self):
        """Discover active OCI models with the TEXT_RERANK capability."""
        if SUPPORTED_OCIGENAI_RERANK_MODELS:
            return

        client_kwargs = CLIENT_KWARGS.copy()
        client_kwargs["service_endpoint"] = (
            f"https://generativeai.{OCI_REGION}.oci.oraclecloud.com"
        )
        generative_ai_client = GenerativeAiClient(**client_kwargs)
        response = generative_ai_client.list_models(
            compartment_id=OCI_COMPARTMENT,
            lifecycle_state="ACTIVE",
        )

        for model in response.data.items:
            on_demand_retired = getattr(model, "time_on_demand_retired", None)
            is_on_demand_available = (
                on_demand_retired is None
                or on_demand_retired > datetime.now(timezone.utc)
            )
            if (
                "TEXT_RERANK" in model.capabilities
                and is_on_demand_available
                and model.display_name == OCI_RERANK_DEFAULT_MODEL
            ):
                model_name = model.display_name
                SUPPORTED_OCIGENAI_RERANK_MODELS[model_name] = {
                    "type": "rerank",
                    "name": model_name,
                    "model_id": model_name,
                    "provider": model_name.split(".")[0] if "." in model_name else "UNKNOWN",
                    "region": OCI_REGION,
                    "compartment_id": OCI_COMPARTMENT,
                }

        logger.info(
            "Successfully loaded %s rerank models from API",
            len(SUPPORTED_OCIGENAI_RERANK_MODELS),
        )

    def list_models(self) -> list[str]:
        return list(SUPPORTED_OCIGENAI_RERANK_MODELS.keys())

    def _invoke_model(self, request: RerankRequest):
        model_id = self.model_id or request.model
        model_info = SUPPORTED_OCIGENAI_RERANK_MODELS[model_id]
        region = model_info["region"]
        self.generative_ai_inference_client.base_client._endpoint = (
            INFERENCE_ENDPOINT_TEMPLATE.replace("{region}", region)
        )
        details = RerankRequestAdapter(model_info).to_oci(request)

        if DEBUG:
            logger.info("OCI Generative AI rerank request: %s", details)

        try:
            response = self.generative_ai_inference_client.rerank_text(
                rerank_text_details=details
            )
            if DEBUG:
                logger.info("OCI Generative AI rerank response: %s", response.data)
            return response.data
        except Exception as exc:
            logger.error("Rerank request failed: %s", exc)
            status_code = getattr(exc, "status", 400)
            if not isinstance(status_code, int) or not 400 <= status_code <= 599:
                status_code = 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    def rerank(self, request: RerankRequest) -> RerankResponse:
        return RerankRequestAdapter.to_cohere(self._invoke_model(request))


def get_rerank_model(model_id: str = None) -> OCIGenAIRerankModel:
    resolved_model_id = model_id or OCI_RERANK_DEFAULT_MODEL
    model = OCIGenAIRerankModel(model_id=resolved_model_id)
    if resolved_model_id in SUPPORTED_OCIGENAI_RERANK_MODELS:
        return model

    logging.error("Unsupported rerank model id %s", model_id)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported rerank model id {resolved_model_id}",
    )
