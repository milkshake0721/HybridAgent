"""HybridAgent: a hybrid reasoning agent that thinks in latent space and acts through text.

Core package shared by HybridSFT, HybridRL and inference: the HybridLink adapter, the
model wrapper and loader, the system prompt, and the hybrid generation engine with its tools.
"""
from .answer import extract_answer, strip_latent_spans
from .backends import (
    SGLangHybridEngine,
    compile_hybridlink_tensorrt,
    enable_tensorrt_acceleration,
    get_sglang_server_command,
    patch_bf16_experts_fastpath,
    prepare_sglang_model_dir,
)
from .constants import ANCHOR_TOKENS, ENTRY_ANCHOR, EXIT_ANCHOR, special_token_id
from .engine import (
    DECODE,
    DONE,
    FEED,
    IDLE,
    LATENT,
    HybridEngine,
    HybridGenerationEngine,
    HybridRound,
    HybridTrajectory,
    RowState,
)
from .hybridlink import HybridLink
from .model import HybridAgentModel, add_anchor_tokens, load_hybrid_agent, resolve_model_dir
from .prompts import SYSTEM_PROMPT
from .readout import LatentSpanReadout, latent_readout, render_with_readout
from .tools import ToolExecutor, build_tool_executor

__all__ = [
    "ANCHOR_TOKENS", "ENTRY_ANCHOR", "EXIT_ANCHOR", "SYSTEM_PROMPT",
    "DECODE", "LATENT", "FEED", "IDLE", "DONE",
    "HybridAgentModel", "HybridEngine", "HybridGenerationEngine", "HybridLink",
    "HybridRound", "HybridTrajectory", "LatentSpanReadout", "RowState",
    "SGLangHybridEngine", "ToolExecutor", "add_anchor_tokens", "build_tool_executor",
    "compile_hybridlink_tensorrt", "enable_tensorrt_acceleration", "extract_answer",
    "get_sglang_server_command", "latent_readout", "load_hybrid_agent",
    "patch_bf16_experts_fastpath", "prepare_sglang_model_dir", "render_with_readout",
    "resolve_model_dir", "special_token_id", "strip_latent_spans",
]
