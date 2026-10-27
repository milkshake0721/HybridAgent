"""HybridRL configuration: the task presets, the fixed recipe constants,
and the few parameters exposed on the command line.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

TASK_PRESETS = {
    "math": dict(
        ability="math",
        max_steps=200,
        max_tool_calls=3,            # tool calls per trajectory
        search_result_length=500,    # characters kept per search snippet
        max_response_tokens=5000,    # per-trajectory budget: text + latent + tool tokens
        min_latent_steps=4,          # steps per latent span
        max_latent_steps=180,
    ),
    "search": dict(
        ability="qa",
        max_steps=150,
        max_tool_calls=6,
        search_result_length=300,
        max_response_tokens=12000,
        min_latent_steps=4,
        max_latent_steps=200,
    ),
}
TASKS = tuple(TASK_PRESETS)

# Rollout
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
MAX_TURNS = 30
MAX_SEQUENCE_TOKENS = 16888      # replay length cap (prompt + response)
PYTHON_TIMEOUT = 120
SEARCH_MAX_RESULTS = 10

# Reward
R_TOOL = 0.1
R_FORMAT_FAIL = -1.0
MAX_ANSWER_CHARS = 1024

# Policy update
CLIP_EPS = 0.2
DUAL_CLIP_C = 3.0
LOG_RATIO_CLIP = 20.0
ADV_EPS = 1e-6
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
MAX_GRAD_NORM = 1.0
LOGIT_CHUNK_SIZE = 256

LOG_ROLLOUTS_PER_STEP = 2
WANDB_PROJECT = "HybridAgent"


@dataclass
class HybridRLConfig:
    model: str
    data: str
    task: str
    output_dir: str
    max_steps: int
    save_steps: int = 20
    prompts_per_step: int = 64
    group_size: int = 8
    num_minibatches: int = 8
    lr: float = 1e-6
    seed: int = 42
    max_tool_calls: int = 10
    max_response_tokens: int = 5000
    min_latent_steps: int = 4
    max_latent_steps: int = 200
    search_result_length: int = 500
    freeze_experts: bool = False
    use_8bit: bool = False
    wandb: bool = False

    @property
    def ability(self) -> str:
        return TASK_PRESETS[self.task]["ability"]

    def validate(self) -> None:
        if self.task not in TASK_PRESETS:
            raise ValueError(f"unknown task {self.task!r}; expected one of {TASKS}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.save_steps < 0:
            raise ValueError("save_steps must be non-negative")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if not 1 <= self.num_minibatches <= self.prompts_per_step:
            raise ValueError("num_minibatches must be in [1, prompts_per_step]")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if not 0 <= self.min_latent_steps <= self.max_latent_steps:
            raise ValueError("need 0 <= min_latent_steps <= max_latent_steps")
        if not 0 < self.max_response_tokens < MAX_SEQUENCE_TOKENS:
            raise ValueError(f"max_response_tokens must be in (0, {MAX_SEQUENCE_TOKENS})")
        if self.search_result_length <= 0:
            raise ValueError("search_result_length must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
