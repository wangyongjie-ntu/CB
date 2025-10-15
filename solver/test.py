import os
import json
import asyncio
# import pytest
from pathlib import Path
from solver import Solver
from storage import NanoVectorDBStorage
from llm import gpt_41_mini_complete, openai_embedding
from utils import compute_mdhash_id
from utils import always_get_an_event_loop


def solver():
    solver = Solver(
        embedding_func=openai_embedding,
        llm_model_func=gpt_41_mini_complete,
        vector_db_storage_cls=NanoVectorDBStorage,
        always_create_working_dir=True,
        max_step=5,
    )
    return solver




loop = always_get_an_event_loop()
question = "What is the capital of France?"
solver = solver()
solver.helper(solver, question)
