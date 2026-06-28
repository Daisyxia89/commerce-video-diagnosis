"""前端消费层输出契约装配层。

把后端内部的商品理解 / 视频理解 / 视频说服诊断模块产物，收敛为前端 5 Tab
可直接消费的标准响应对象（《电商短视频诊断：前端消费层输出契约》SSOT）。
"""
from commerce_video_diagnosis.understanding.assembly.response_assembler import (
    assemble_frontend_response,
)

__all__ = ["assemble_frontend_response"]
