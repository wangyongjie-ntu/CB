#Filename:	tools.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 02:48:01 PM CST
#Description: This files contains functions that call external tools and collect the response.
from tooluniverse import ToolUniverse
import nano_vectordb
import json
import bm25s
import llm
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from jinja2 import Template

class Tool:
    def __init__(self, name, description, func_call):
        self.name = name
        self.description = description
        self.func_call = func_call
        self.tooluniverse = ToolUniverse(tool_files=self.tool_files_dict)
        self.tooluniverse.load_tools()

    def rag_infer(self, query, top_k=5):
        return self.rag_model.rag_infer(query, top_k)

    def tool_RAG(self, message=None, picked_tool_names=None, rag_num=5,return_call_result=False):
        extra_factor = 30  # Factor to retrieve more than rag_num
        if picked_tool_names is None:
            assert picked_tool_names is not None or message is not None
            picked_tool_names = self.rag_infer(
                message, top_k=rag_num*extra_factor)

        picked_tool_names_no_special = []
        for tool in picked_tool_names:
            if tool not in self.special_tools_name:
                picked_tool_names_no_special.append(tool)
        picked_tool_names_no_special = picked_tool_names_no_special[:rag_num]
        picked_tool_names = picked_tool_names_no_special[:rag_num]

        picked_tools = self.tooluniverse.get_tool_by_name(picked_tool_names)
        picked_tools_prompt = self.tooluniverse.prepare_tool_prompts(
            picked_tools)
        if return_call_result:
            return picked_tools_prompt, picked_tool_names
        return picked_tools_prompt # P_RAG

    def generate_fcall_str(user_question: str, conversation_history: List[Dict[str, str]], llm: LLM, tokenizer,
            system_prompt: str = "You are a helpful assistant that solves problems through step-by-step reasoning and tool use. Use the provided functions in JSON format: [{\"name\": \"tool_name\", \"arguments\": {\"key\": value}}]",
            temperature: float = 0.3,
            max_new_tokens: int = 1024,
            top_k_rag: int = 5
    ) -> str:
        relevant_tool_names = rag_infer(user_question, top_k=top_k_rag)
        relevant_tools = ToolUniverse.get_tool_by_name(relevant_tool_names)
        tool_descriptions = ToolUniverse.prepare_tool_prompts(relevant_tools)

        tools_prompt = "\nAvailable tools:\n" + json.dumps(tool_descriptions, indent=2)

        full_conversation = [{"role": "system", "content": system_prompt + tools_prompt}] + conversation_history
        full_conversation.append({"role": "user", "content": user_question})

        chat_template = Template(tokenizer.chat_template)
        prompt = chat_template.render(messages=full_conversation)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=["</s>", "\n\n"],
            skip_special_tokens=True
        )

        outputs = llm.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        if generated_text.startswith("[") and generated_text.endswith("]"):
            return generated_text
        else:
            return ""

    def call_tool(self, fcall_str, return_message=True):
        function_call_json, message = self.tooluniverse.extract_function_call_json(
            fcall_str, return_message=return_message, verbose=True
        )
        call_results = []
        if function_call_json is not None and isinstance(function_call_json, list):
            for func in function_call_json:
                print(f"Tool Call: {func}")
                call_result = self.tooluniverse.run_one_function(func)

                # 记录工具调用结果（含call_id用于追踪）
                call_id = self.tooluniverse.call_id_gen()
                func["call_id"] = call_id
                call_results.append({
                    "role": "tool",
                    "content": json.dumps({"content": call_result, "call_id": call_id})
                })

        # 整理调用结果，返回给对话历史
        revised_messages = [{
            "role": "assistant",
            "content": message.strip(),
            "tool_calls": json.dumps(function_call_json)
        }] + call_results
        return revised_messages

