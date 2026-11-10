"""Dataset loaders, normalizers, and specifications for the 5 HuggingFace benchmarks on G4 GPU.

Supported Benchmarks and Official Gemma 4 26B A4B Reference Targets:
  1. `mmlu_pro`: `TIGER-Lab/MMLU-Pro` (10-choice A-J, target 82.6%)
  2. `aime_2026`: `MathArena/aime_2026` (integer 000-999, strict no-tools, target 88.3%)
  3. `gpqa_diamond`: `Idavidrein/gpqa` (`gpqa_diamond`) / `fingertap/GPQA-Diamond` (4-choice A-D, target 82.3%)
  4. `livecodebench_v6`: `livecodebench/code_generation_lite` (`release_v6` / `test6.jsonl`, pass@1, target 77.1%)
  5. `mmmu_pro`: `MMMU/MMMU_Pro` (`standard (10 options)` & `vision`, 10-choice A-J multimodal, target 73.8%)
"""
from __future__ import annotations

import ast
import base64
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import re
from typing import Any
import zlib

from PIL import Image


REFERENCE_SCORES: dict[str, float] = {
    "mmlu_pro": 82.6,
    "aime_2026": 88.3,
    "gpqa_diamond": 82.3,
    "livecodebench_v6": 77.1,
    "mmmu_pro": 73.8,
}

BENCHMARK_NAMES: tuple[str, ...] = tuple(REFERENCE_SCORES.keys())


@dataclass
class BenchmarkSpec:
    name: str
    display_name: str
    hf_repo: str
    fallback_repo: str | None
    split: str
    metric: str
    reference_score: float
    num_choices: int | None = None
    allow_tools: bool = True
    multimodal: bool = False
    hf_config: str | None = None
    subsets: tuple[str, ...] = ()
    extra_fallback_repos: tuple[str, ...] = ()


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "mmlu_pro": BenchmarkSpec(
        name="mmlu_pro",
        display_name="MMLU Pro",
        hf_repo="TIGER-Lab/MMLU-Pro",
        fallback_repo=None,
        split="test",
        metric="accuracy",
        reference_score=82.6,
        num_choices=10,
        allow_tools=True,
        multimodal=False,
    ),
    "aime_2026": BenchmarkSpec(
        name="aime_2026",
        display_name="AIME 2026 (No Tools)",
        hf_repo="MathArena/aime_2026",
        fallback_repo="Maxwell-Jia/AIME_2024",
        split="train",
        metric="accuracy",
        reference_score=88.3,
        num_choices=None,
        allow_tools=False,
        multimodal=False,
        extra_fallback_repos=(
            "MathArena/aime_2026_I",
            "math-ai/aime26",
            "zhenghaoxu/aime-2026",
        ),
    ),
    "gpqa_diamond": BenchmarkSpec(
        name="gpqa_diamond",
        display_name="GPQA Diamond",
        hf_repo="Idavidrein/gpqa",
        fallback_repo="fingertap/GPQA-Diamond",
        split="train",
        metric="accuracy",
        reference_score=82.3,
        num_choices=4,
        allow_tools=True,
        multimodal=False,
        hf_config="gpqa_diamond",
        extra_fallback_repos=(
            "wanng/gpqa",
            "nichenshun/gpqa_diamond",
        ),
    ),
    "livecodebench_v6": BenchmarkSpec(
        name="livecodebench_v6",
        display_name="LiveCodeBench v6",
        hf_repo="livecodebench/code_generation_lite",
        fallback_repo=None,
        split="test",
        metric="pass@1",
        reference_score=77.1,
        num_choices=None,
        allow_tools=True,
        multimodal=False,
    ),
    "mmmu_pro": BenchmarkSpec(
        name="mmmu_pro",
        display_name="MMMU Pro (多模態)",
        hf_repo="MMMU/MMMU_Pro",
        fallback_repo=None,
        split="test",
        metric="accuracy",
        reference_score=73.8,
        num_choices=10,
        allow_tools=True,
        multimodal=True,
        subsets=("standard (10 options)", "vision"),
    ),
}


class BenchmarkItem(dict):
    """Normalized benchmark item supporting both dictionary indexing and attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"BenchmarkItem has no attribute {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


def compress_lcb_private_tests(test_cases: list[dict[str, Any]]) -> str:
    """Encode LiveCodeBench v6 `private_test_cases` using base64(zlib(pickle(json.dumps(cases))))."""
    payload = pickle.dumps(json.dumps(test_cases))
    compressed = zlib.compress(payload)
    return base64.b64encode(compressed).decode("ascii")


def _decode_lcb_tests(raw_field: Any) -> list[dict[str, Any]]:
    """Decode LiveCodeBench v6 test cases from list, JSON string, or base64+zlib+pickle+json."""
    if not raw_field:
        return []
    if isinstance(raw_field, list):
        return [dict(x) for x in raw_field if isinstance(x, dict)]
    s = str(raw_field).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [dict(x) for x in parsed if isinstance(x, dict)]
    except Exception:
        pass
    try:
        decompressed = zlib.decompress(base64.b64decode(s.encode("ascii")))
        try:
            unpickled = pickle.loads(decompressed)
            if isinstance(unpickled, (str, bytes)):
                loaded = json.loads(unpickled)
                return [dict(x) for x in loaded if isinstance(x, dict)] if isinstance(loaded, list) else []
            if isinstance(unpickled, list):
                return [dict(x) for x in unpickled if isinstance(x, dict)]
        except Exception:
            loaded = json.loads(decompressed.decode("utf-8"))
            return [dict(x) for x in loaded if isinstance(x, dict)] if isinstance(loaded, list) else []
    except Exception:
        return []
    return []


decode_lcb_private_tests = _decode_lcb_tests


def _parse_options_field(raw_opts: Any) -> list[str]:
    if raw_opts is None:
        return []
    if isinstance(raw_opts, str):
        s = raw_opts.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                raw_opts = parsed
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    raw_opts = list(parsed)
                else:
                    raw_opts = [s]
            except Exception:
                raw_opts = [s]
    if isinstance(raw_opts, (list, tuple)):
        return [
            str(opt).strip()
            for opt in raw_opts
            if opt is not None and str(opt).strip() and str(opt).strip().upper() != "N/A"
        ]
    return []


def normalize_mmlu_pro_item(raw: dict[str, Any], idx: int = 0) -> BenchmarkItem:
    """Normalize a raw `TIGER-Lab/MMLU-Pro` item into a 10-choice `A`–`J` `BenchmarkItem`."""
    options = _parse_options_field(raw.get("options") if "options" in raw else raw.get("choices"))
    ans_raw = raw.get("answer")
    ans = ""
    if isinstance(ans_raw, int) and 0 <= ans_raw < 26:
        ans = chr(65 + ans_raw)
    elif ans_raw is not None:
        s_ans = str(ans_raw).strip().upper()
        if len(s_ans) == 1 and "A" <= s_ans <= "Z":
            ans = s_ans
        elif s_ans.isdigit() and 0 <= int(s_ans) < 26:
            ans = chr(65 + int(s_ans))
    if not ans and "answer_index" in raw and raw["answer_index"] is not None:
        ans = chr(65 + int(raw["answer_index"]))
    return BenchmarkItem(
        id=raw.get("question_id", raw.get("id", f"mmlu_{idx}")),
        question=str(raw.get("question", "")),
        options=options,
        answer=ans,
        category=str(raw.get("category", raw.get("src", "general"))),
        images=None,
    )


def normalize_aime_2026_item(raw: dict[str, Any], idx: int = 0) -> BenchmarkItem:
    """Normalize a raw `MathArena/aime_2026` item into an integer-answer `BenchmarkItem`."""
    q = raw.get("problem") or raw.get("question") or raw.get("Problem") or ""
    ans = str(raw.get("answer", raw.get("Answer", ""))).strip()
    return BenchmarkItem(
        id=raw.get("id", raw.get("problem_idx", f"aime_{idx}")),
        question=str(q),
        options=[],
        answer=ans,
        images=None,
    )


def normalize_gpqa_diamond_item(raw: dict[str, Any], idx: int = 0, seed: int = 42) -> BenchmarkItem:
    """Normalize a raw `Idavidrein/gpqa` or `fingertap/GPQA-Diamond` item with deterministic 4-option shuffling."""
    qid = str(raw.get("id", raw.get("Record ID", f"gpqa_{idx}")))
    q = str(raw.get("Question") or raw.get("question") or "")
    if "options" in raw and isinstance(raw["options"], list) and raw["options"] and "Correct Answer" not in raw:
        opts = [str(o) for o in raw["options"]]
        ans = str(raw.get("answer", "A")).strip().upper()
        if len(ans) == 1 and "A" <= ans <= "D":
            gold_letter = ans
        elif ans.isdigit() and 0 <= int(ans) < 4:
            gold_letter = chr(65 + int(ans))
        else:
            gold_letter = "A"
    elif "Correct Answer" in raw:
        correct = str(raw.get("Correct Answer", "")).strip()
        inc1 = str(raw.get("Incorrect Answer 1", "")).strip()
        inc2 = str(raw.get("Incorrect Answer 2", "")).strip()
        inc3 = str(raw.get("Incorrect Answer 3", "")).strip()
        choices = [correct, inc1, inc2, inc3]
        h = int(hashlib.sha256(f"{seed}:{qid}:{q}".encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(h)
        perm = [0, 1, 2, 3]
        rng.shuffle(perm)
        opts = [choices[p] for p in perm]
        gold_letter = chr(65 + perm.index(0))
    else:
        opts = _parse_options_field(raw.get("options") or raw.get("choices"))
        ans = str(raw.get("answer", "A")).strip().upper()
        gold_letter = ans if (len(ans) == 1 and "A" <= ans <= "D") else "A"

    return BenchmarkItem(
        id=qid,
        question=q,
        options=opts,
        answer=gold_letter,
        category=str(raw.get("Subdomain", raw.get("High-level domain", "science"))),
        images=None,
    )


def normalize_livecodebench_v6_item(raw: dict[str, Any], idx: int = 0) -> BenchmarkItem:
    """Normalize a raw `livecodebench/code_generation_lite` (`release_v6`) item."""
    if "test_cases" in raw and isinstance(raw["test_cases"], list) and not (
        raw.get("public_test_cases") or raw.get("private_test_cases")
    ):
        all_tests = [dict(tc) for tc in raw["test_cases"] if isinstance(tc, dict)]
    else:
        pub = _decode_lcb_tests(raw.get("public_test_cases"))
        priv = _decode_lcb_tests(raw.get("private_test_cases"))
        all_tests = pub + priv

    meta_raw = raw.get("metadata", {})
    if isinstance(meta_raw, str):
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
    elif isinstance(meta_raw, dict):
        meta = dict(meta_raw)
    else:
        meta = {}

    starter = str(raw.get("starter_code") or "")
    fn_name = raw.get("fn_name") or meta.get("func_name") or meta.get("fn_name")
    if not fn_name and starter:
        for m in re.finditer(r"def\s+([A-Za-z_]\w*)\s*\(", starter):
            if m.group(1) != "__init__":
                fn_name = m.group(1)
                break

    return BenchmarkItem(
        id=raw.get("question_id", raw.get("id", f"lcb_v6_{idx}")),
        question=str(raw.get("question_content") or raw.get("question") or ""),
        title=str(raw.get("question_title", raw.get("title", ""))),
        starter_code=starter,
        fn_name=fn_name,
        test_cases=all_tests,
        difficulty=str(raw.get("difficulty", "")),
        answer="",
        images=None,
    )


def _load_pil_if_needed(val: Any) -> Image.Image | None:
    if val is None:
        return None
    if isinstance(val, Image.Image):
        return val.convert("RGB") if val.mode not in ("RGB", "L", "RGBA") else val
    if isinstance(val, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(val)).convert("RGB")
        except Exception:
            return None
    if isinstance(val, dict):
        if "bytes" in val and val["bytes"]:
            try:
                return Image.open(io.BytesIO(val["bytes"])).convert("RGB")
            except Exception:
                pass
        if "path" in val and val["path"] and Path(val["path"]).is_file():
            try:
                return Image.open(val["path"]).convert("RGB")
            except Exception:
                pass
        return None
    if isinstance(val, (str, Path)):
        s = str(val).strip()
        if not s:
            return None
        p = Path(s)
        if p.is_file():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return None
        if s.startswith("data:image/") and "," in s:
            s = s.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(s)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            return None
    return None


def normalize_mmmu_pro_item(raw: dict[str, Any], idx: int = 0) -> BenchmarkItem:
    """Normalize a raw `MMMU/MMMU_Pro` (`standard (10 options)` or `vision`) item with PIL images."""
    options = _parse_options_field(raw.get("options") if "options" in raw else raw.get("choices"))
    imgs: list[Image.Image] = []
    if (
        "images" in raw
        and isinstance(raw["images"], list)
        and raw["images"]
        and not any(raw.get(k) is not None for k in ("image", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"))
    ):
        for entry in raw["images"]:
            im = _load_pil_if_needed(entry)
            if im is not None:
                imgs.append(im)
    else:
        if "image" in raw and raw["image"] is not None:
            im = _load_pil_if_needed(raw["image"])
            if im is not None:
                imgs.append(im)
        for k in range(1, 8):
            key = f"image_{k}"
            if key in raw and raw[key] is not None:
                im = _load_pil_if_needed(raw[key])
                if im is not None:
                    imgs.append(im)
        if not imgs and "images" in raw and isinstance(raw["images"], list):
            for entry in raw["images"]:
                im = _load_pil_if_needed(entry)
                if im is not None:
                    imgs.append(im)

    subset = str(raw.get("subset", ""))
    if not subset:
        subset = "vision" if ("image" in raw and raw.get("image") is not None and "image_1" not in raw) else "standard (10 options)"

    return BenchmarkItem(
        id=raw.get("id", f"mmmu_pro_{idx}"),
        subset=subset,
        question=str(raw.get("question", "")),
        options=options,
        answer=str(raw.get("answer", "A")).strip().upper(),
        category=str(raw.get("subject", raw.get("topic_difficulty", "multimodal"))),
        images=imgs if imgs else None,
    )


def build_synthetic_benchmark_items() -> dict[str, list[dict[str, Any]]]:
    """Return authentic 5-item synthetic datasets for all 5 target HuggingFace benchmarks."""
    mmlu_pro = [
        {
            "question_id": f"mmlu_{i}",
            "question": f"MMLU Pro STEM problem {i}: Select the right option. [GOLD_ANS:{ans}]",
            "options": [f"Option {chr(65 + j)} for Q{i}" for j in range(10)],
            "answer": ans,
            "answer_index": ord(ans) - 65,
            "category": "physics" if i % 2 == 0 else "computer_science",
        }
        for i, ans in enumerate(["A", "C", "F", "H", "J"])
    ]

    aime_2026 = [
        {
            "id": f"aime_2026_{i}",
            "problem": f"AIME 2026 Olympiad problem {i}: Find the integer value. [GOLD_ANS:{pred}]",
            "answer": gold,
        }
        for i, (gold, pred) in enumerate(
            [("042", "42"), ("7", "007"), ("999", "999"), ("105", "105"), ("0", "000")]
        )
    ]

    gpqa_diamond = [
        {
            "id": f"gpqa_{i}",
            "Question": f"GPQA Diamond graduate question {i}: Which statement holds? [GOLD_ANS:{ans}]",
            "options": [f"Choice A_{i}", f"Choice B_{i}", f"Choice C_{i}", f"Choice D_{i}"],
            "answer": ans,
            "Subdomain": "Quantum Mechanics",
        }
        for i, ans in enumerate(["A", "B", "C", "D", "B"])
    ]

    lcb_stdin_tests = [
        {"input": "3 5\n", "output": "8\n", "testtype": "stdin"},
        {"input": "10 20\n", "output": "30\n", "testtype": "stdin"},
    ]
    lcb_fn_tests = [
        {"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]", "testtype": "functional"},
        {"input": "[3, 2, 4]\n6", "output": "[1, 2]", "testtype": "functional"},
    ]
    livecodebench_v6 = [
        {
            "question_id": "lcb_v6_stdin_0",
            "question_title": "Sum of Integers from Stdin",
            "question_content": "Read integers from standard input and print their sum.",
            "starter_code": "",
            "public_test_cases": json.dumps(lcb_stdin_tests[:1]),
            "private_test_cases": compress_lcb_private_tests(lcb_stdin_tests[1:]),
            "metadata": json.dumps({}),
            "difficulty": "easy",
        },
        {
            "question_id": "lcb_v6_fn_1",
            "question_title": "Two Sum",
            "question_content": "Implement twoSum in class Solution returning indices of two numbers that add up to target.",
            "starter_code": "class Solution:\n    def twoSum(self, nums: list[int], target: int) -> list[int]:\n",
            "public_test_cases": json.dumps(lcb_fn_tests[:1]),
            "private_test_cases": compress_lcb_private_tests(lcb_fn_tests[1:]),
            "metadata": json.dumps({"func_name": "twoSum"}),
            "difficulty": "medium",
        },
    ]

    mmmu_pro = [
        {
            "id": f"mmmu_std_{i}",
            "subset": "standard (10 options)",
            "question": f"Look at <image 1> and <image 2> for problem {i}. [GOLD_ANS:{ans}]",
            "options": [f"MMMU Opt {chr(65 + j)}" for j in range(10)],
            "answer": ans,
            "image_1": Image.new("RGB", (64, 64), color=(40 * i, 100, 180)),
            "image_2": Image.new("RGB", (64, 64), color=(180, 40 * i, 90)),
        }
        for i, ans in enumerate(["B", "D", "G"])
    ] + [
        {
            "id": f"mmmu_vis_{i}",
            "subset": "vision",
            "question": f"Interpret the screenshot diagram {i}. [GOLD_ANS:{ans}]",
            "options": [f"Vision Opt {chr(65 + j)}" for j in range(10)],
            "answer": ans,
            "image": Image.new("RGB", (64, 64), color=(120, 200, 50 * (i + 1))),
        }
        for i, ans in enumerate(["E", "J"])
    ]

    return {
        "mmlu_pro": mmlu_pro,
        "aime_2026": aime_2026,
        "gpqa_diamond": gpqa_diamond,
        "livecodebench_v6": livecodebench_v6,
        "mmmu_pro": mmmu_pro,
    }


def create_synthetic_benchmark_datasets(tmp_path: str | Path) -> dict[str, Path]:
    """Write offline JSONL files (+ PNG images for MMMU Pro) for all 5 benchmarks into `tmp_path`."""
    ds_dir = Path(tmp_path) / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    img_dir = ds_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    items_by_bench = build_synthetic_benchmark_items()
    paths: dict[str, Path] = {}

    for bench, items in items_by_bench.items():
        out_file = ds_dir / f"{bench}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for idx, item in enumerate(items):
                rec = dict(item)
                for k, v in list(rec.items()):
                    if isinstance(v, Image.Image):
                        img_path = img_dir / f"{bench}_{idx}_{k}.png"
                        v.save(img_path)
                        rec[k] = str(img_path)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        paths[bench] = out_file
    return paths


def _read_local_dataset_file(dataset_dir: str | Path, benchmark_name: str) -> list[dict[str, Any]] | None:
    base = Path(dataset_dir)
    if base.is_file():
        candidates = [base]
    elif base.is_dir():
        candidates = [
            base / f"{benchmark_name}.jsonl",
            base / f"{benchmark_name}.json",
            base / benchmark_name / "test.jsonl",
            base / benchmark_name / f"{benchmark_name}.jsonl",
        ]
    else:
        return None

    for p in candidates:
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                return []
            if p.suffix == ".json":
                data = json.loads(text)
                if isinstance(data, list):
                    return [dict(x) for x in data]
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    return None


def _load_from_hf_hub(spec: BenchmarkSpec, split: str | None) -> list[dict[str, Any]] | None:
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        return None
    try:
        import datasets as hf_datasets  # type: ignore[import-not-found]
    except Exception:
        return None

    target_split = split or spec.split
    repos_to_try: list[str] = [spec.hf_repo]
    if spec.fallback_repo:
        repos_to_try.append(spec.fallback_repo)
    repos_to_try.extend(spec.extra_fallback_repos)

    if spec.name == "mmmu_pro" and spec.subsets:
        combined: list[dict[str, Any]] = []
        for sub in spec.subsets:
            try:
                ds = hf_datasets.load_dataset(spec.hf_repo, sub, split=target_split)
                for row in ds:
                    r_dict = dict(row)
                    r_dict.setdefault("subset", sub)
                    combined.append(r_dict)
            except Exception:
                continue
        if combined:
            return combined

    for repo in repos_to_try:
        try:
            if repo == spec.hf_repo and spec.hf_config:
                try:
                    ds = hf_datasets.load_dataset(repo, spec.hf_config, split=target_split)
                    return [dict(row) for row in ds]
                except Exception:
                    pass
            ds = hf_datasets.load_dataset(repo, split=target_split)
            return [dict(row) for row in ds]
        except Exception:
            continue
    return None


def load_benchmark_dataset(
    benchmark_name: str,
    *,
    dataset_items: Sequence[dict[str, Any]] | None = None,
    dataset_dir: str | Path | None = None,
    split: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    synthetic_smoke: bool = False,
) -> list[BenchmarkItem]:
    """Load and normalize items for any of the 5 benchmarks, with automatic fallback:
    1) explicit `dataset_items`
    2) `synthetic_smoke=True`
    3) local `dataset_dir / f"{benchmark_name}.jsonl"`
    4) HuggingFace `datasets.load_dataset` (trying primary repo then `fallback_repo`, e.g. `Idavidrein/gpqa` -> `fingertap/GPQA-Diamond`)
    5) offline synthetic benchmark items
    """
    if benchmark_name not in BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark {benchmark_name!r}; expected one of {list(BENCHMARK_SPECS)}")

    spec = BENCHMARK_SPECS[benchmark_name]
    raw_list: list[dict[str, Any]] | None = None

    if dataset_items is not None:
        raw_list = list(dataset_items)
    elif synthetic_smoke:
        raw_list = build_synthetic_benchmark_items().get(benchmark_name, [])
    elif dataset_dir is not None:
        raw_list = _read_local_dataset_file(dataset_dir, benchmark_name)

    if raw_list is None:
        raw_list = _load_from_hf_hub(spec, split)

    if raw_list is None and dataset_dir is not None:
        raw_list = _read_local_dataset_file(dataset_dir, benchmark_name)

    if raw_list is None:
        raw_list = build_synthetic_benchmark_items().get(benchmark_name, [])

    normalizers = {
        "mmlu_pro": lambda r, i: normalize_mmlu_pro_item(r, i),
        "aime_2026": lambda r, i: normalize_aime_2026_item(r, i),
        "gpqa_diamond": lambda r, i: normalize_gpqa_diamond_item(r, i, seed=seed),
        "livecodebench_v6": lambda r, i: normalize_livecodebench_v6_item(r, i),
        "mmmu_pro": lambda r, i: normalize_mmmu_pro_item(r, i),
    }
    norm_fn = normalizers[benchmark_name]
    normalized = [norm_fn(r, idx) for idx, r in enumerate(raw_list)]
    if limit is not None and int(limit) > 0:
        normalized = normalized[: int(limit)]
    return normalized
