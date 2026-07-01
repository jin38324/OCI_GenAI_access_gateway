from oci.generative_ai_inference import models as oci_models

from api.schema import RerankRequest, RerankResponse, RerankResult


class RerankRequestAdapter:
    def __init__(self, model_info: dict):
        self.model_info = model_info

    def _serving_mode(self):
        endpoint_id = self.model_info.get("endpoint")
        if endpoint_id:
            return oci_models.DedicatedServingMode(
                serving_type="DEDICATED",
                endpoint_id=endpoint_id,
            )
        return oci_models.OnDemandServingMode(
            serving_type="ON_DEMAND",
            model_id=self.model_info["model_id"],
        )

    def to_oci(self, request: RerankRequest) -> oci_models.RerankTextDetails:
        details = {
            "input": request.query,
            "compartment_id": self.model_info["compartment_id"],
            "serving_mode": self._serving_mode(),
            "documents": request.documents,
            "is_echo": False,
        }
        if request.top_n is not None:
            details["top_n"] = request.top_n
        if request.max_tokens_per_doc is not None:
            details["max_tokens_per_document"] = request.max_tokens_per_doc
        return oci_models.RerankTextDetails(**details)

    @staticmethod
    def to_cohere(response: oci_models.RerankTextResult) -> RerankResponse:
        return RerankResponse(
            id=response.id,
            results=[
                RerankResult(
                    index=document_rank.index,
                    relevance_score=document_rank.relevance_score,
                )
                for document_rank in response.document_ranks
            ],
        )
