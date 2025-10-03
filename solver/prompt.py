#Filename:	prompt.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 02:47:30 PM CST

PROMPTS = {}
GRAPH_FIELD_SEP = "<SEP>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["helper"] = """
Please act as a reasoning assistant that provides hints for the next step in solving the clinical/medical question. Your role is to guide the reasoning process by suggesting what to consider or explore next. **Do not** give the final answer, **nor** provide information that directly reveals it. Only generate hints that point toward a single, immediate reasoning step.

Given Input:
Question: {question}
Reasoning: {reasoning}

Next reasoning step(hints only): 
Output: 
"""

PROMPTS["init_tools"] = """


Given Input:
Question: {question}
Tool lists: {list_of_tools}

Selected Tools (seperated by comma):
Output:
"""

PROMPTS["solver"] = """
You must fully understand and solve a question through reasoning and function calls.

Guidelines:
• For each step, you must generate a reasoning thought and correct function call. If needed, call multiple functions.
• If you think you have answered the question, thoroughly reflect on your reasoning to verify you have in fact answered the question. If not, continue reasoning. If so, call the ‘Finish’ function and provide your final answer, which should be 1) comprehensive, 2) explain how you arrived at the answer, and 3) why the answer addresses the question.
• If the result from the last function call is empty or not useful, you must continue reasoning and call ToolRAG (or simulate a virtual ToolRAG call) to retrieve more tools.
    – If the tool you need is in the Function List below, you must retrieve them using a
    virtual ToolRAG call that simulates obtaining the tool through ToolRAG.
    – If the tool you need is not in the Function List below, you need to call ToolRAG.
    – {Description of ToolRAG and virtual ToolRAG tools}
• Do not answer the question based on general knowledge. You must answer the question based on the information returned by the tools.
• If all previous solution attempts have failed, do not repeat the same thoughts and function calls. Instead, come up with new solution approaches.

Given Input: 
Function List: {description}
Reasoning step (json): {reasoning step format}
Previous reasoning steps: {previous multi-step reasoning trace}
Hint for next step: {hint}
For the final step, respond in this JSON format, providing the final answer and a detailed
"""

PROMPTS["reflection"] = """
xxxx
"""
