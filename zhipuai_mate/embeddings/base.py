from typing import List, Any, Optional

import zhipuai
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from pydantic.v1 import root_validator
from zhipuai.types.embeddings import EmbeddingsResponded


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    client: Any
    model: str = "embedding-2"
    zhipuai_api_key: Optional[str] = None

    @root_validator(pre=True)
    def build_client(cls, values: dict) -> dict:
        if not values.get('client'):
            values['client'] = zhipuai.ZhipuAI(
                api_key=values.get('zhipuai_api_key')
            )
        return values

    def _embed_text(self, text) -> List[float]:
        response: EmbeddingsResponded = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        data = response.data
        return data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            text = text.replace("\n", " ")
            results.append(self._embed_text(text))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)
