import logging
import json
import re
import ast
from typing import Any
from pydantic import BaseModel

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def element_to_dict(element):
    if isinstance(element, dict):
        return element
    elif isinstance(element, BaseModel):
        return element.model_dump_json()
    else:
        return json.loads(str(element))


def trim_image(obj: Any,
                head: int = 10,
                tail: int = 10,
                placeholder: str = "...<trimmed>..."
                ) -> str:    
    # print(type(obj))
    # print(obj)
    if isinstance(obj,dict):
        serializable = obj
    else:
        try:
            serializable = obj.dict()
        except Exception:
            try:           
                s = obj.strip()
                try:
                    serializable = json.loads(s)
                except Exception:
                    serializable = ast.literal_eval(s)
            
            except Exception:
                try:
                    serializable = json.loads(str(obj))
                except Exception:
                    serializable = str(obj)
    json_str = json.dumps(serializable, ensure_ascii=False, indent=2)
    

    if "image_url" in json_str:
        # 截断函数
        def _trim_b64(b64: str, head: int, tail: int, placeholder: str) -> str:
            if len(b64) <= head + tail + len(placeholder):
                return b64
            return b64[:head] + placeholder + b64[-tail:]

        # 替换函数
        def repl(m: re.Match) -> str:
            prefix = m.group(1)
            b64data = m.group(2)
            trimmed = _trim_b64(b64data, head=head, tail=tail, placeholder=placeholder)
            return prefix + trimmed

        DATAURI_RE = re.compile(
            r'(data:image\/[A-Za-z0-9.+-]+;base64,)'   # group1 = prefix (mime + base64,)
            r'([A-Za-z0-9+/=\n\r]+?)'                 # group2 = base64 data (non-greedy)
            r'(?=")',                                 # stop at the JSON closing quote
            flags=re.IGNORECASE | re.DOTALL
        )

        sanitized = DATAURI_RE.sub(repl, json_str)
        return sanitized

    else:
        return json_str

