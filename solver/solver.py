#Filename:	solver.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 02:48:15 PM CST

import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
from prompt import *

from .base import (
    StorageNameSpace,
    BaseVectorStorage,
    BaseLexicalStorage
)

from .storage import (
    NanoVectorDBStorage,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    always_get_an_event_loop,
    logger,
)

from .llm import (
    gpt_4o_complete,
    gpt_41_mini_complete,
    openai_embedding,
    vllm_local_complete,
    vllm_local_embedding,
)

@dataclass 
class Solver():
    # text embedding
    working_dir = "storage",
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 64
    embedding_func_max_async: int = 32
    consine_better_than_threshold = 0.6

    # LLM
    llm_model_func: callable = gpt_41_mini_complete
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 8

    #Storage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    
    # Other arguments
    max_step = 10
    always_create_working_dir = True
    addon_params: dict = field(default_factory=dict)

    def __post_init__():
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"Cure Bench Project initial param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        
        self.tooluniverse_vdb: BaseVectorStorage = (
            self.vector_db_storage_cls(
                namespace="tooluniverse",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields = {}
            )
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            self.llm_model_func
        )


    async def helper(question: str, reasoning: list[str]) -> str:

        return
    
    async def thoughts() -> str:
        return
    
    async def toolrag(question: str) -> list(str):

        return 

    async def generate(question: str):
        for i in range(self.max_step):


        return

