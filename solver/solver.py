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

# from tool import Tool
from .prompt import PROMPTS
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


def normalize_to_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        parts = []
        for item in x:
            txt = normalize_to_text(item)
            if txt:
                parts.append(txt)
        return "\n".join(parts)
    if isinstance(x, dict):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    # 其它类型（数值、None、自定义对象）兜底
    return str(x)


@dataclass
class Solver():
    # namespace: str = "tooluniverse"
    working_dir: str = "storage"
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 64
    embedding_func_max_async: int = 32
    cosine_better_than_threshold: float = 0

    llm_model_func: callable = gpt_41_mini_complete
    llm_model_name: str = ""
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 8

    # Storage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    
    # Other arguments
    max_step: int = 5
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)

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
        return await self.llm_model_func(
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format if response_format else {}
        )


    async def helper(self, question: str, reasoning: list[str]) -> str:
        prompt_template = PROMPTS["helper"]
        context = dict(
            question=question,
            reasoning=";".join(reasoning)
        )
        user_prompt = prompt_template.format(**context)
        results = await self.llm_model_func(
            prompt= user_prompt,

        )

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
            list_of_tools=";".join(tool_descriptions),
            selected_tools_format='{"tools": ["..."]}'
        )
        use_prompt = prompt_template.format(**context)
        results = await self.use_llm_func(use_prompt, response_format={"type": "json_object"})
        print(results)
        data = json.loads(results)
        tools = data.get("tools", [])
        try:
            data = json.loads(results)
            tools = data.get("tools", [])
            return tools[:20]
        except Exception as e:
            logger.error(f"Init Tools Error: {str(e)}")
            return []

    async def toolrag(self, reasoning: str) -> list[str]:
        try:
            results = await self.tooluniverse_vdb.query(reasoning, top_k=5)
            print("ToolRAG Results: ", results)
            return [result.get('name') for result in results if 'name' in result]
        except Exception as e:
            logger.error(f"ToolRAG Search Error: {str(e)}")
            return []

    async def judge_tools(self, question: str, hint: str, retrieved_content: str) -> Optional[bool]:
        prompt_template = PROMPTS["judge_tool"]
        context = dict(
            question=question,
            hint=hint,
            retrieved_content=retrieved_content,
            judge_format='{"judge": <bool>}'
        )
        prompt = prompt_template.format(**context)
        results = await self.use_llm_func(prompt, response_format={"type": "json_object"})

        data = json.loads(results)
        val = data.get("judge")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if val_lower in ("true", "yes", "1"):
                return True
            elif val_lower in ("false", "no", "0"):
                return False

    async def extract_useful_info(self, question: str, hint: str, retrieved_content: str) -> str:
        prompt_template = PROMPTS["extract_useful_info"]
        context = dict(
            question=question,
            hint=hint,
            retrieved_content=retrieved_content
        )
        prompt = prompt_template.format(**context)
        results = await self.use_llm_func(prompt, response_format={"type": "text"})

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

        filtered_tools = []
        for name in function_list:
            tools = Tooluniverse.get_tool_by_name([name])
            if tools:
                filtered_tools.append(tools[0])
                print("Filtered Tool Add: ", tools[0])

        tool_descriptions = Tooluniverse.prepare_tool_prompts(filtered_tools)
        print("Tool Descriptions: ", tool_descriptions)

        context = dict(
            description=tool_descriptions,
            reasoning_step_format=json.dumps(
                {
                    "reasoning": "...",
                    "function_calls": [
                        {"name": "...", "arguments": {"param": "..."}}
                    ],
                    "final_answer": ""
                },
            ),
            previous_multi_step_reasoning_trace=";".join(prev_reason),
            hint=hint,
            question=question,
            Description_of_ToolRAG_and_virtual_ToolRAG_tools="""ToolRAG retrieves relevant tools from the tool universe. 
Virtual ToolRAG simulates retrieving tools that are already in the Function List.""",
            final_answer_format=json.dumps(
                {
                    "reasoning": "...",
                    "function_calls": [
                        {"name": "...", "arguments": {"param": "..."}}
                    ],
                    "final_answer": ""
                },
            ),
        )
        prompt = prompt_template.format(**context)
        result = await self.use_llm_func(prompt, response_format={"type": "json_object"})
        print("\033[31mSOLVER Result: \033[0m", result)
        # try:
        data = json.loads(result)
        reasoning = data.get("reasoning", "")
        tools = data.get("function_calls", [])
        print("Extracted Tools: ", tools)
        fcall_str = json.dumps(tools, ensure_ascii=False, indent=2) if tools else "[]"
        return reasoning, fcall_str
        # except json.JSONDecodeError:
        #     logger.warning(f"无法解析LLM返回的JSON: {result}")
        #     return result, []

    async def generate(self, question: str) -> tuple[list[str], str]:
        reasoning_trace: list[str] = []
        tool_trace: list[str] = []

        print("Init Tools: ")
        init_tools = await self.init_tools(question)

        trigger = 0
        final_answer = ""
        Tooluniverse = tooluniverse.ToolUniverse()
        Tooluniverse.load_tools()
        
        for i in range(self.max_step):
            hint = await self.helper(question, reasoning_trace)
            print(f"Step {i+1} Hint: ", hint)

            current_tools = list(set(init_tools + tool_trace))
            print("Current Tools: ", current_tools)
            reasoning, fcall_str = await self.thoughts(current_tools, question, reasoning_trace, hint)
            print(f"\033[34mStep {i+1} Reasoning: \033[0m", reasoning)
            print(f"\033[31mStep {i+1} Tool Call: \033[0m", fcall_str)

            reasoning_trace.append(reasoning)
            has_finish = any(
                isinstance(c, dict) and str(c.get("name", "")).lower() == "finish"
                for c in getattr(self, "last_function_calls", [])
            )
            if has_finish:
                final_answer = reasoning
                break

            retrieved_content = ""
            
            if fcall_str:
                function_call_json = Tooluniverse.extract_function_call_json(
                    fcall_str,
                    verbose=True
                )
            

            call_results = []
            if function_call_json is not None and isinstance(function_call_json, list):
                for func in function_call_json:
                    print(f"Tool Call: {func}")
                    call_result = Tooluniverse.run_one_function(func)
                    print(f"Tool Call Result: {call_result}")
                    call_id = Tooluniverse.call_id_gen()
                    func["call_id"] = call_id
                    call_results.append({
                        "role": "tool",
                        "content": json.dumps({"content": call_result, "call_id": call_id})
                    })
            revised_messages = [{
                "role": "assistant",
                # "content": message.strip(),
                "tool_calls": json.dumps(function_call_json)
            }] + call_results
            retrieved_content += normalize_to_text(revised_messages)

            if retrieved_content:
                flag = await self.judge_tools(question, hint, retrieved_content)
                print(f"Step {i+1} Judge Tool Result: ", flag)

                data = json.loads(fcall_str)
                toolist = [item['name'] for item in data]
                tool_trace.extend(toolist)
                if flag:
                    useful_info = await self.extract_useful_info(question, hint, retrieved_content)
                    print(f"Step {i+1} Useful Info: ", useful_info)
                    reasoning_trace.append(f"useful info: {useful_info}")
                    trigger = 0
                else:
                    trigger += 1

                if trigger > 1:
                    new_tools = await self.toolrag(" ".join(reasoning_trace))
                    print("New Tools from ToolRAG: ", new_tools)
                    init_tools = list(set(init_tools + new_tools))
                    trigger = 0

        if not final_answer:
            final_answer = f"In {self.max_step} step, get Answer: " + "; ".join(reasoning_trace[-3:])

        return reasoning_trace, final_answer



if __name__ == "__main__":
    solver = Solver()
    quetion = ""
    file_path = r"/home/yongjie/Kunhong/CureBench/data/curebench_valset_pharse1.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        q = first_line.get("question", "")
        opts = first_line.get("options", {})
        question = q.strip() + "\n" + "\n".join([f"{k}. {v}" for k, v in opts.items()])
        print(question)
    loop = always_get_an_event_loop()
    response = loop.run_until_complete(solver.generate(question))
    print("response: ", response)
