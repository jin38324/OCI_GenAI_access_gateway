import os
import sys
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from api.routers import rerank as rerank_router
from api.schema import RerankResponse, RerankResult


class InProcessTransport(httpx.BaseTransport):
    """Let the Cohere SDK call a FastAPI TestClient without a live server."""

    def __init__(self, client: TestClient):
        self.client = client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.client.request(
            request.method,
            request.url.path,
            headers=dict(request.headers),
            content=request.content,
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )


class RerankApiTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(rerank_router.router, prefix="/v2")
        self.client = TestClient(app)
        self.payload = {
            "query": "capital of the United States",
            "documents": ["Carson City", "Washington, D.C."],
            "top_n": 1,
        }
        self.cohere_payload = self.payload | {"model": "cohere.rerank-v4.0-fast"}
        self.model = Mock()
        self.model.rerank.return_value = RerankResponse(
            id="rerank-id",
            results=[RerankResult(index=1, relevance_score=0.99)],
        )

    @patch("api.routers.rerank.get_rerank_model")
    def test_returns_cohere_v2_response(self, get_rerank_model):
        get_rerank_model.return_value = self.model

        response = self.client.post(
            "/v2/rerank",
            headers={"Authorization": "Bearer ocigenerativeai"},
            json=self.cohere_payload,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "id": "rerank-id",
                "results": [{"index": 1, "relevance_score": 0.99}],
                "meta": {
                    "api_version": {"version": "2", "is_experimental": False}
                },
            },
        )
        request = self.model.rerank.call_args.args[0]
        self.assertEqual(request.max_tokens_per_doc, 4096)

    @patch("api.routers.rerank.get_rerank_model")
    def test_open_webui_v1_request_contract(self, get_rerank_model):
        get_rerank_model.return_value = self.model
        app = FastAPI()
        app.include_router(rerank_router.v1_router, prefix="/v1")
        client = TestClient(app)

        response = client.post(
            "/v1/rerank",
            headers={"Authorization": "Bearer ocigenerativeai"},
            json=self.cohere_payload,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"results": [{"index": 1, "relevance_score": 0.99}]},
        )
        get_rerank_model.assert_called_once_with("cohere.rerank-v4.0-fast")

    @patch("api.routers.rerank.get_rerank_model")
    def test_v1_direct_request_can_omit_model(self, get_rerank_model):
        get_rerank_model.return_value = self.model
        app = FastAPI()
        app.include_router(rerank_router.v1_router, prefix="/v1")
        client = TestClient(app)

        response = client.post(
            "/v1/rerank",
            headers={"Authorization": "Bearer ocigenerativeai"},
            json=self.payload,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"results": [{"index": 1, "relevance_score": 0.99}]},
        )
        get_rerank_model.assert_called_once_with(None)

    def test_requires_bearer_authentication(self):
        response = self.client.post("/v2/rerank", json=self.payload)

        self.assertIn(response.status_code, (401, 403))

    def test_validates_cohere_request(self):
        response = self.client.post(
            "/v2/rerank",
            headers={"Authorization": "Bearer ocigenerativeai"},
            json={"query": "query", "documents": []},
        )

        self.assertEqual(response.status_code, 422)

    @patch("api.routers.rerank.get_rerank_model")
    def test_propagates_unsupported_model_error(self, get_rerank_model):
        get_rerank_model.side_effect = HTTPException(
            status_code=400,
            detail="Unsupported rerank model id unknown",
        )

        response = self.client.post(
            "/v2/rerank",
            headers={"Authorization": "Bearer ocigenerativeai"},
            json=self.payload | {"model": "unknown"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported rerank model id unknown")

    @patch("api.routers.rerank.get_rerank_model")
    def test_cohere_client_v2_compatibility(self, get_rerank_model):
        import cohere

        get_rerank_model.return_value = self.model
        http_client = httpx.Client(transport=InProcessTransport(self.client))
        cohere_client = cohere.ClientV2(
            api_key="ocigenerativeai",
            base_url="http://testserver",
            httpx_client=http_client,
        )

        response = cohere_client.rerank(
            model="cohere.rerank-v4.0-fast",
            query="capital of the United States",
            documents=["Carson City", "Washington, D.C."],
            top_n=1,
        )

        self.assertEqual(response.id, "rerank-id")
        self.assertEqual(response.results[0].index, 1)
        self.assertEqual(response.results[0].relevance_score, 0.99)
        self.assertEqual(response.meta.api_version.version, "2")


if __name__ == "__main__":
    unittest.main()
