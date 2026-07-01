import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from api.models import oci_rerank
from api.schema import RerankRequest


class OCIGenAIRerankModelTest(unittest.TestCase):
    def setUp(self):
        self.models = oci_rerank.SUPPORTED_OCIGENAI_RERANK_MODELS
        self.original_models = self.models.copy()
        self.models.clear()

    def tearDown(self):
        self.models.clear()
        self.models.update(self.original_models)

    @patch("api.models.oci_rerank.GenerativeAiInferenceClient")
    @patch("api.models.oci_rerank.GenerativeAiClient")
    def test_discovers_only_text_rerank_models(self, management_client, inference_client):
        management_client.return_value.list_models.return_value = SimpleNamespace(
            data=SimpleNamespace(
                items=[
                    SimpleNamespace(
                        display_name="cohere.rerank-v4.0-fast",
                        capabilities=["TEXT_RERANK"],
                    ),
                    SimpleNamespace(
                        display_name="cohere.embed-v4.0",
                        capabilities=["TEXT_EMBEDDINGS"],
                    ),
                    SimpleNamespace(
                        display_name="cohere.rerank-v5.0",
                        capabilities=["TEXT_RERANK"],
                    ),
                    SimpleNamespace(
                        display_name="cohere.rerank-v3.5",
                        capabilities=["TEXT_RERANK"],
                        time_on_demand_retired=datetime(2025, 5, 1, tzinfo=timezone.utc),
                    ),
                ]
            )
        )

        model = oci_rerank.OCIGenAIRerankModel()

        self.assertEqual(model.list_models(), ["cohere.rerank-v4.0-fast"])
        self.assertEqual(
            self.models["cohere.rerank-v4.0-fast"]["model_id"],
            "cohere.rerank-v4.0-fast",
        )
        management_client.return_value.list_models.assert_called_once_with(
            compartment_id=oci_rerank.OCI_COMPARTMENT,
            lifecycle_state="ACTIVE",
        )
        inference_client.assert_called_once()

    def test_rerank_registry_is_separate_from_chat_and_embedding(self):
        from api import setting

        self.assertIsNot(
            setting.SUPPORTED_OCIGENAI_RERANK_MODELS,
            setting.SUPPORTED_OCIGENAI_CHAT_MODELS,
        )
        self.assertIsNot(
            setting.SUPPORTED_OCIGENAI_RERANK_MODELS,
            setting.SUPPORTED_OCIGENAI_EMBEDDING_MODELS,
        )
    @patch("api.models.oci_rerank.GenerativeAiInferenceClient")
    def test_calls_oci_rerank_text_and_converts_response(self, inference_client):
        self.models["cohere.rerank-v4.0-fast"] = {
            "type": "rerank",
            "name": "cohere.rerank-v4.0-fast",
            "model_id": "cohere.rerank-v4.0-fast",
            "provider": "cohere",
            "region": "us-chicago-1",
            "compartment_id": "ocid1.compartment.oc1..example",
        }
        rank = SimpleNamespace(index=1, relevance_score=0.98)
        inference_client.return_value.rerank_text.return_value = SimpleNamespace(
            data=SimpleNamespace(id="rerank-id", document_ranks=[rank])
        )
        request = RerankRequest(
            model="cohere.rerank-v4.0-fast",
            query="query",
            documents=["first", "second"],
            top_n=1,
        )

        response = oci_rerank.OCIGenAIRerankModel().rerank(request)

        self.assertEqual(response.id, "rerank-id")
        self.assertEqual(response.results[0].index, 1)
        call = inference_client.return_value.rerank_text.call_args
        details = call.kwargs["rerank_text_details"]
        self.assertEqual(details.input, "query")
        self.assertEqual(details.documents, ["first", "second"])
        self.assertEqual(details.top_n, 1)

    @patch("api.models.oci_rerank.GenerativeAiInferenceClient")
    def test_rejects_other_reranker_model_names(self, inference_client):
        self.models["cohere.rerank-v4.0-fast"] = {
            "model_id": "cohere.rerank-v4.0-fast",
            "region": "us-chicago-1",
            "compartment_id": "ocid1.compartment.oc1..example",
        }

        with self.assertRaises(HTTPException) as raised:
            oci_rerank.get_rerank_model("BAAI/bge-reranker-v2-m3")

        self.assertEqual(raised.exception.status_code, 400)

    @patch("api.models.oci_rerank.GenerativeAiInferenceClient")
    def test_uses_default_model_when_request_omits_model(self, inference_client):
        self.models["cohere.rerank-v4.0-fast"] = {
            "model_id": "cohere.rerank-v4.0-fast",
            "region": "us-chicago-1",
            "compartment_id": "ocid1.compartment.oc1..example",
        }

        model = oci_rerank.get_rerank_model()

        self.assertEqual(model.model_id, "cohere.rerank-v4.0-fast")

    @patch("api.models.oci_rerank.GenerativeAiInferenceClient")
    def test_preserves_oci_http_error_status(self, inference_client):
        self.models["cohere.rerank-v4.0-fast"] = {
            "model_id": "cohere.rerank-v4.0-fast",
            "region": "us-chicago-1",
            "compartment_id": "ocid1.compartment.oc1..example",
        }
        error = RuntimeError("rate limited")
        error.status = 429
        inference_client.return_value.rerank_text.side_effect = error
        request = RerankRequest(
            model="cohere.rerank-v4.0-fast",
            query="query",
            documents=["document"],
        )

        with self.assertRaises(HTTPException) as raised:
            oci_rerank.OCIGenAIRerankModel().rerank(request)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertIn("rate limited", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
