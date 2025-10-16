#Filename:	retriever.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 02:42:53 PM CST

import bm25s
import numpy as np
from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar
from .utils import EmbeddingFunc

T = TypeVar("T")

@dataclass
class StorageNameSpace:
    global_config: dict
    namespace: str

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass

@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)
    # working_dir: str = "./storage"

    async def query(self, query:str, top_k:int) -> list[str]:
        raise NotImplementedError

    async def upsert(self, data:list[str]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

@dataclass
class BaseLexicalStorage(StorageNameSpace):

    async def query(self, query:str, top_k:int) -> list[str]:
        raise NotImplementedError

    async def upsert(self, data:list[str]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

