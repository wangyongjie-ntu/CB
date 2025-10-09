# Filename:	solver.py
# Author:	Wang Yongjie
# Email:	yongjie.wang@ntu.edu.sg
# Date:		Thu 02 Oct 2025 02:48:15 PM CST

import asyncio
import re
import os
import json
import tooluniverse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial, wraps
from typing import Callable, Dict, List, Optional, Type, Union, cast

from prompt import PROMPTS
from base import (
    StorageNameSpace,
    BaseVectorStorage,
    BaseLexicalStorage
)
from storage import (
    NanoVectorDBStorage,
)
from utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    always_get_an_event_loop,
    logger,
)
from llm import (
    gpt_4o_complete,
    gpt_41_mini_complete,
    openai_embedding,
    vllm_local_complete,
    vllm_local_embedding,
)


@dataclass
class Solver():
    # text embedding
    working_dir = "storage"
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 64
    embedding_func_max_async: int = 32
    cosine_better_than_threshold: float = 0.6

    llm_model_func: callable = vllm_local_complete
    llm_model_name: str = "llama-3.1-8b"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 8

    # Storage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage

    # Other arguments
    max_step: int = 10
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"Cure Bench Project initial param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)

        self.tooluniverse_vdb: BaseVectorStorage = (
            self.vector_db_storage_cls(
                namespace="tooluniverse",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields=set()
            )
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(self.llm_model_func)

    async def use_llm_func(self, prompt: str, system_prompt: Optional[str] = None,
                           response_format: Optional[dict] = None) -> str:
        try:
            return await self.llm_model_func(
                prompt=prompt,
                system_prompt=system_prompt,
                response_format=response_format if response_format else {}
            )
        except Exception as e:
            logger.error(f"LLM调用出错: {str(e)}")
            return ""

    async def helper(self, question: str, reasoning: list[str]) -> str:
        # generate the hints for the next step.
        prompt_template = PROMPTS["helper"]
        context = dict(
            question=question,
            reasoning=";".join(reasoning)
        )
        use_prompt = prompt_template.format(**context)
        results = await self.use_llm_func(use_prompt)  # response_format={"type": "json_object"})

        # Parse results
        pattern = re.compile(r'output:\s*(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(results)

        if match:
            clean_result = match.group(1).strip()
            return clean_result
        else:
            return results

    async def init_tools(self, question: str) -> list[str]:
        Tooluniverse = tooluniverse.ToolUniverse()
        Tooluniverse.load_tools()
        all_tools = Tooluniverse.prepare_tool_prompts(Tooluniverse.all_tools)
        tool_descriptions = [f"{tool['name']}: {tool['description']}" for tool in all_tools]

        prompt_template = PROMPTS["init_tools"]
        context = dict(
            question=question,
            list_of_tools=";".join(tool_descriptions)
        )
        use_prompt = prompt_template.format(**context)
        results = await self.use_llm_func(use_prompt)

        pattern = re.compile(r'Selected Tools.*?:\s*(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(results)

        if match:
            tools_str = match.group(1).strip()
            tools = [tool.strip() for tool in tools_str.split(',')]
            return tools[:20]
        return []

    async def toolrag(self, reasoning: str) -> list[str]:
        try:
            results = await self.tooluniverse_vdb.query(reasoning, top_k=5)
            return [result.get('name') for result in results if 'name' in result]
        except Exception as e:
            logger.error(f"ToolRAG Search Error: {str(e)}")
            return []

    async def judge_tools(self, question: str, hint: str, retrieved_content: str) -> Optional[bool]:
        # Levarge LLM to judge whether retrieved content match the question.
        prompt_template = PROMPTS["judge_tool"]
        context = dict(
            question=question,
            hint=hint,
            retrieved_content=retrieved_content
        )
        prompt = prompt_template.format(**context)
        results = await self.use_llm_func(prompt)

        pattern = re.compile(r'output:\s*(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(results)

        flag = None
        if match:
            output_content = match.group(1).strip()
            bool_match = re.search(r'\b(true|false)\b', output_content, re.IGNORECASE)
            if bool_match:
                flag = bool_match.group(1).lower() == 'true'
            else:
                logger.warning(f"无法提取: {output_content}")

        assert flag is None or isinstance(flag, bool), "flag必须是布尔值或None"
        return flag

    async def extract_useful_info(self, question: str, hint: str, retrieved_content: str) -> str:
        # Leverage LLM to keep the useful information.
        prompt_template = PROMPTS["extract_useful_info"]
        context = dict(
            question=question,
            hint=hint,
            retrieved_content=retrieved_content
        )
        prompt = prompt_template.format(**context)
        results = await self.use_llm_func(prompt)

        pattern = re.compile(r'output:\s*(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(results)

        if match:
            return match.group(1).strip()
        return results.strip()

    async def thoughts(self, function_list: list[str], question: str, prev_reason: list[str], hint: str):
        # Leverage LLM to provide potential tools
        prompt_template = PROMPTS["solver"]

        Tooluniverse = tooluniverse.ToolUniverse()
        Tooluniverse.load_tools()
        tool_descriptions = []
        for tool_name in function_list:
            tool = Tooluniverse.get_tool_by_name([tool_name])
            if tool:
                tool_descriptions.append(f"{tool[0]['name']}: {tool[0]['description']}")

        context = dict(
            description=";".join(tool_descriptions),
            reasoning_step_format='{"reasoning": "...", "tools": ["..."]}',
            previous_multi_step_reasoning_trace=";".join(prev_reason),
            hint=hint,
            question=question,
            Description_of_ToolRAG_and_virtual_ToolRAG_tools="""ToolRAG retrieves relevant tools from the tool universe. 
Virtual ToolRAG simulates retrieving tools that are already in the Function List."""
        )
        prompt = prompt_template.format(**context)
        result = await self.use_llm_func(prompt, response_format={"type": "json_object"})
        try:
            result_json = json.loads(result)
            return result_json.get("reasoning", ""), result_json.get("tools", [])
        except json.JSONDecodeError:
            logger.warning(f"无法解析LLM返回的JSON: {result}")
            return result, []

    async def generate(self, question: str) -> tuple[list[str], str]:
        reasoning_trace: list[str] = []
        tool_trace: list[str] = []
        init_tools = await self.init_tools(question)
        trigger = 0
        final_answer = ""

        for i in range(self.max_step):
            hint = await self.helper(question, reasoning_trace)
            current_tools = list(set(init_tools + tool_trace))
            reasoning, toolist = await self.thoughts(current_tools, question, reasoning_trace, hint)


            reasoning_trace.append(reasoning)

            if toolist and any(tool.lower() == "finish" for tool in toolist):
                final_answer = reasoning
                break

            retrieved_content = ""
            Tooluniverse = tooluniverse.ToolUniverse()
            Tooluniverse.load_tools()
            for tool in toolist:
                try:
                    func_call = [{"name": tool, "parameters": {}}]
                    fcall_str = json.dumps(func_call)
                    function_call_json, _ = Tooluniverse.extract_function_call_json(fcall_str)

                    if function_call_json and isinstance(function_call_json, list):
                        for func in function_call_json:
                            result = Tooluniverse.run_one_function(func)
                            retrieved_content += f"\nTool {tool} Call Result: {str(result)}"
                    else:
                        retrieved_content += f"\nTool {tool} Format Error."
                except Exception as e:
                    retrieved_content += f"\nTool {tool} Execute Error: {str(e)}"

            # 判断工具结果是否有用
            if retrieved_content:
                flag = await self.judge_tools(question, hint, retrieved_content)

                if flag:
                    useful_info = await self.extract_useful_info(question, hint, retrieved_content)
                    reasoning_trace.append(f"useful info: {useful_info}")
                    tool_trace.extend(toolist)
                    trigger = 0
                else:
                    tool_trace.extend(toolist)
                    trigger += 1

                if trigger > 1:
                    new_tools = await self.toolrag(" ".join(reasoning_trace))
                    init_tools = list(set(init_tools + new_tools))
                    trigger = 0

        if not final_answer:
            final_answer = f"In {self.max_step} step, get Answer: " + "; ".join(reasoning_trace[-3:])

        return reasoning_trace, final_answer
