#Filename:	storage.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 02:59:23 PM CST

import os
import json
import asyncio
import numpy as np
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Union, cast
from nano_vectordb import NanoVectorDB
from .base import (
    StorageNameSpace,
    BaseVectorStorage,
    BaseLexicalStorage
)
from .utils import (
    logger,
    set_logger,
    compute_mdhash_id
)

@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.1
    def __post_init__(self):
        print(self.global_config)
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_tooluniverse.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: list[dict]):

        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = []
        contents = []
        for line in data:
            list_data.append({"__id__":compute_mdhash_id(json.dumps(line)), **line})
            remove_keys = [k for k in line.keys() if k not in self.meta_fields]  # collect first
            for k in remove_keys:
                line.pop(k, None)

            contents.append(json.dumps(line))

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k = 5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        # print("\033[31mQuery Embedding: \033[0m",embedding)
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def isin(self, entity_name:str):
        try:
            if self.client.get(entity_name):
                return True
            else:
                return False
        finally:
            return False

    async def delete_entity(self, entity_name: str):
        try:
            if self._client.get(entity_name):
                self._client.delete(entity_name)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def index_done_callback(self):
        self._client.save()

