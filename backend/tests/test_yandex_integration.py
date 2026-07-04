import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.llm import get_llm, _should_retry_yandex_without_structured_output
from src.make_relationships import create_chunk_vector_index
from src.shared.common_fn import load_embedding_model
from src.yandex_embeddings import YandexEmbeddings


class FakeEmbeddingsEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.25] * kwargs["dimensions"])]
        )


class YandexEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = FakeEmbeddingsEndpoint()
        self.embeddings = YandexEmbeddings(
            api_key="test-key",
            folder_id="test-folder",
            doc_model="emb://folder/doc/latest",
            query_model="emb://folder/query/latest",
            dimensions=4,
            client=SimpleNamespace(embeddings=self.endpoint),
        )

    def test_document_and_query_models_are_kept_separate(self):
        documents = self.embeddings.embed_documents([" first\n document ", "second"])
        query = self.embeddings.embed_query(" search\nquery ")

        self.assertEqual([4, 4], [len(vector) for vector in documents])
        self.assertEqual(4, len(query))
        self.assertEqual(
            ["emb://folder/doc/latest", "emb://folder/doc/latest", "emb://folder/query/latest"],
            [call["model"] for call in self.endpoint.calls],
        )
        self.assertEqual(["first document", "second", "search query"], [call["input"] for call in self.endpoint.calls])
        self.assertTrue(all(call["dimensions"] == 4 for call in self.endpoint.calls))

    def test_empty_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            self.embeddings.embed_query(" \n ")


class YandexGraphBuilderIntegrationTests(unittest.TestCase):
    @patch("src.shared.common_fn.YandexEmbeddings.from_env")
    def test_embedding_loader_accepts_yandex_provider(self, from_env):
        from_env.return_value = SimpleNamespace(dimensions=768)
        embeddings, dimension = load_embedding_model("yandex", "yandex-text-embeddings-v2")

        self.assertIs(embeddings, from_env.return_value)
        self.assertEqual(768, dimension)

    @patch("src.llm.ChatOpenAI")
    def test_yandex_llm_uses_openai_compatible_endpoint_and_project_header(self, chat_openai):
        fake_llm = MagicMock()
        chat_openai.return_value = fake_llm
        settings = {
            "LLM_MODEL": "gpt://folder/aliceai-llm/latest",
            "YANDEX_API_KEY": "test-key",
            "YANDEX_FOLDER_ID": "folder",
            "YANDEX_BASE_URL": "https://ai.api.cloud.yandex.net/v1",
        }

        with patch.dict(os.environ, settings, clear=False):
            llm, model_name, _ = get_llm("yandex_aliceai")

        self.assertIs(llm, fake_llm)
        self.assertEqual(settings["LLM_MODEL"], model_name)
        kwargs = chat_openai.call_args.kwargs
        self.assertEqual(settings["LLM_MODEL"], kwargs["model"])
        self.assertEqual(settings["YANDEX_BASE_URL"], kwargs["base_url"])
        self.assertEqual({"OpenAI-Project": "folder"}, kwargs["default_headers"])

    def test_only_yandex_capability_errors_trigger_unstructured_retry(self):
        yandex_llm = SimpleNamespace(_graph_builder_provider="yandex")
        other_llm = SimpleNamespace(_graph_builder_provider="other")
        bad_request = SimpleNamespace(status_code=400)

        self.assertTrue(_should_retry_yandex_without_structured_output(yandex_llm, bad_request))
        self.assertFalse(_should_retry_yandex_without_structured_output(other_llm, bad_request))

    @patch("src.make_relationships.Neo4jVector")
    @patch("src.make_relationships.load_embedding_model")
    @patch("src.make_relationships.execute_graph_query")
    def test_mismatched_vector_index_is_recreated(self, execute_query, load_model, neo4j_vector):
        embedding = object()
        load_model.return_value = (embedding, 768)
        execute_query.side_effect = [[{"name": "vector", "dimensions": 384}], None]

        create_chunk_vector_index(MagicMock(), "yandex", "yandex-text-embeddings-v2")

        self.assertIn("DROP INDEX vector", execute_query.call_args_list[1].args[1])
        self.assertEqual(768, neo4j_vector.call_args.kwargs["embedding_dimension"])
        neo4j_vector.return_value.create_new_index.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
