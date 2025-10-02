import asyncio
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os
from .utils import wrap_embedding_func_with_attrs

global_openai_async_client = None
global_vllm_async_client = None

def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client

def get_vllm_async_client_instance(base_url:str = None, api_key:str = None):
    global global_vllm_async_client
    if global_vllm_async_client is None:
        global_vllm_async_client = AsyncOpenAI( base_url = base_url, api_key = api_key)
    return global_vllm_async_client

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:

    openai_async_client = get_openai_async_client_instance()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs)

    return response.choices[0].message.content

async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_41_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4.1-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

### VLLM server 
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def vllm_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:

    vllm_openai_client = get_vllm_async_client_instance(base_url = "http://localhost:8000/v1", api_key = "EMPTY")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    response = await vllm_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs)
    return response.choices[0].message.content

async def vllm_local_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await vllm_openai_complete_if_cache(
        "Qwen3-8B",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@wrap_embedding_func_with_attrs(embedding_dim=1000, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def vllm_local_embedding(texts: list[str]) -> np.ndarray:
    vllm_embed_client = get_vllm_async_client_instance(base_url="http://localhost:8000/v1", api_key = "EMPTY")
    response = await vllm_embed_client.embeddings.create(
        model="bge-m3", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
