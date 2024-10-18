import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def async_stream_chat(
    self,
    tokenizer,
    ins: str,
    his: List[Dict[str, Any]] = [],
    max_length: int = 4096,
    top_p: float = 0.7,
    temperature: float = 0.9,
    **kwargs,
):
    # ... 前面的代码保持不变 ...

    try:
        start_time = time.monotonic()
        response = await asyncfy_with_semaphore(
            lambda: self.client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p, **extra_params
            )
        )()
        
        # 添加对 response 的类型检查
        if isinstance(response, str):
            logger.error(f"Unexpected response type: {response}")
            raise ValueError(f"API返回了意外的字符串响应: {response}")
        
        generated_text = response.choices[0].message.content
        generated_tokens_count = response.usage.completion_tokens
        input_tokens_count = response.usage.prompt_tokens
        time_cost = time.monotonic() - start_time
        # ... 后面的代码保持不变 ...
    except Exception as e:
        logger.error(f"Error in async_stream_chat: {e}")
        logger.error(f"Response type: {type(response)}")
        logger.error(f"Response content: {response}")
        raise
