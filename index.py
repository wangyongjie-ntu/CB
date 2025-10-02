#Filename:	index.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Thu 02 Oct 2025 03:42:15 PM CST

from tooluniverse import ToolUniverse
import json
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from solver.storage import *
from solver.utils import wrap_embedding_func_with_attrs, always_get_an_event_loop

EMBED_MODEL = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B", device="cpu"
)

# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

storage =  NanoVectorDBStorage(namespace = "tooluniverse", embedding_func = local_embedding, meta_fields = {}, global_config = {"working_dir": "solver/storage", "embedding_batch_num":32})
loop = always_get_an_event_loop()

if __name__ == "__main__":
    toolpool = ToolUniverse()
    toolpool.load_tools()
    #print("load tool description embedding: ")
    #tool_name, _ = toolpool.refresh_tool_name_desc(enable_full_desc=True)
    #print(tool_name)
    all_tools_str = [each for each in toolpool.prepare_tool_prompts(toolpool.all_tools)]
    print(all_tools_str)
    print(all_tools_str[0].keys())
    loop.run_until_complete(storage.upsert(all_tools_str))
    loop.run_until_complete(storage.index_done_callback())
