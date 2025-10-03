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
        # generate the hints for the next step.

        prompt_template = PROMPTS["helper"]
        context = dict(
            question = question,
            reasoning=";".join(reasoning)
        )
        use_prompt = prompt_template.format(**context_base)
        results = await use_llm_func(use_prompt) # response_format={"type": "json_object"})
        
        # Parse results
        pattern = re.compile(r'output:\s*(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(results)

        if match:
            clean_result = match.group(1).strip()
            return clean_result
        else:
            return results
    
    async def init_tools(question: str) -> list[str]:
        # select top-20 tools from all 215 tools.

        prompt_template = PROMPTS["init_thought"]
        context = dict(
            question = question,
            tools=";".join(reasoning)
        )
        use_prompt = prompt_template.format(**context_base)
        results = await use_llm_func(use_prompt) # response_format={"type": "json_object"})
        
        # Parse results
        return tools

    async def toolrag(reasoning)-> list(str):
        ktools = rag.query(reasoning)
        return ktools

    async def judge_tools(question, hint, retrieved_content):
        # Levarge LLM to judge whether retrieved content match the question.

        return True

    async def extract_useful_info(question, hint, retrieved_content):
        # Leverage LLM to keep the useful information.
        prompt + LLM
        return summary

    async def thoughts(function_list: list[str], question:str, prev_reason:str, hint:str):
        # Leverage LLM to provide potential tools

        return reasoning, avail_tools
    
    async def generate(question: str):
        reasoning_trace = {}
        tool_trace = {}
        init_tools = await self.init_tools(question)
        trigger = 0

        for i in range(self.max_step):
            hint = await helper(question, reasoning_trace) # produce hints
            # produce tools and reasoning
            reasoning_trace, toolist = await self.thoughts(question, reasoning_trace, hint, init_tools)

            if finish in toolist:
                # call the finish tool and generate the final answers.
                return reasoning, answer
            else:
                retrieved_content = ""
                for tool in toolist:
                    content = toolcall()
                    retrieved_content += content

                flag = await self.judge_tools(question, hint, retrieved_content)

                if flag: 
                    tool_trace = tool_trace + toolist
                    useinfo = await self.extract_useful_info(question, hint, retrieved_content)
                    reasoning = reasoning + useinfo
                else:
                    tool_trace = tool_trace + toolist # avoid produce the same toolist
                    trigger += 1

                if trigger > 1:
                    new_tools = await self.toolrag(reasoning_trace, tooluniverse)
                    init_tools = init_tools + new_tools
                else:
                    tool_trace = tool_trace # remove preivous unuseful tools
                    continue

        return reasoning, answer




        return

