#Filename:	main.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 02:41:13 PM CST

import json
import asyncio
from solver.utils import always_get_an_event_loop
from solver.solver import Solver

if __name__ == "__main__":
    solver = Solver()
    question = ""
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
