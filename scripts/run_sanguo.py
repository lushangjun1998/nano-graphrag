from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BOOK_PATH = PROJECT_ROOT / "books" / "三国演义.txt"
CACHE_DIRS = {
    "sample": PROJECT_ROOT / "nano_graphrag_cache_sanguo_sample",
    "full": PROJECT_ROOT / "nano_graphrag_cache_sanguo_full",
}
CHAPTER_PATTERN = re.compile(
    r"(?m)^[ \t\u3000]*第[一二三四五六七八九十百千万零〇两]+回[ \t\u3000]"
)
DEFAULT_QUESTIONS = {
    "local": "曹操为什么在赤壁之战中失败？",
    "naive": "桃园结义的誓词原文是什么？",
    "global": "全书如何表现“天下大势，分久必合，合久必分”？",
}


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def read_book(path: Path = BOOK_PATH) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"书籍文件不存在: {path}")
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"书籍文件为空: {path}")
    return text


def split_book_into_chapters(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    starts = [match.start() for match in CHAPTER_PATTERN.finditer(normalized)]
    if not starts:
        raise ValueError("未找到“第 X 回”章节标题")

    preface = normalized[: starts[0]].strip()
    chapters = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(normalized)
        chapter = normalized[start:end].strip()
        if index == 0 and preface:
            chapter = f"{preface}\n\n{chapter}"
        chapters.append(chapter)
    return chapters


def documents_for_scope(scope: str) -> list[str]:
    chapters = split_book_into_chapters(read_book())
    if scope == "sample":
        return chapters[:1]
    if scope == "full":
        return chapters
    raise ValueError(f"未知构建范围: {scope}")


def format_settings_summary(settings: Any) -> str:
    return "\n".join(
        [
            "配置预检（API Key 已隐藏）",
            f"  DeepSeek API Key: {'已配置' if settings.deepseek_api_key else '未配置'}",
            f"  DeepSeek Base URL: {settings.deepseek_base_url}",
            f"  DeepSeek Best Model: {settings.deepseek_best_model}",
            f"  DeepSeek Cheap Model: {settings.deepseek_cheap_model}",
            f"  DeepSeek Thinking: {settings.deepseek_thinking}",
            f"  BGE-M3 模型目录: {settings.bge_m3_model_path}",
            f"  Embedding 设备: {settings.embedding_device}",
        ]
    )


def validate_build_target(
    scope: str,
    cache_dir: Path,
    confirmed: bool,
) -> str:
    if scope == "full" and not confirmed:
        raise ValueError("整书构建会消耗大量 DeepSeek 调用，必须显式传入 --yes")

    if not cache_dir.exists():
        return f"缓存目录尚不存在，构建时创建: {cache_dir}"

    entries = list(cache_dir.iterdir())
    if entries:
        return (
            f"缓存目录已有 {len(entries)} 项，将按现有数据继续或恢复；"
            f"脚本不会删除任何文件: {cache_dir}"
        )
    return f"缓存目录为空，将直接使用: {cache_dir}"


def graphrag_options(cache_dir: Path) -> dict[str, Any]:
    return {
        "working_dir": str(cache_dir),
        "enable_naive_rag": True,
        "best_model_max_async": 4,
        "cheap_model_max_async": 4,
    }


def insert_documents(
    cache_dir: Path,
    documents: Sequence[str],
    graphrag_factory: Callable[..., Any] | None = None,
) -> Any:
    if not documents:
        raise ValueError("没有可插入的章节")
    if graphrag_factory is None:
        from nano_graphrag.graphrag import GraphRAG

        graphrag_factory = GraphRAG

    graph = graphrag_factory(**graphrag_options(cache_dir))
    graph.insert(list(documents))
    return graph


def require_cache_for_mode(cache_dir: Path, mode: str) -> None:
    common = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_entities.json",
        "graph_chunk_entity_relation.graphml",
    ]
    mode_specific = {
        "local": [],
        "global": ["kv_store_community_reports.json"],
        "naive": ["vdb_chunks.json"],
    }
    if mode not in mode_specific:
        raise ValueError(f"未知查询模式: {mode}")

    missing = [
        filename
        for filename in common + mode_specific[mode]
        if not (cache_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"缓存不完整，无法执行 {mode} 查询。缺少: {', '.join(missing)}"
        )


def _json_length(path: Path, data_key: str | None = None) -> int | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data_key is not None:
        data = data.get(data_key)
    if data is None:
        return None
    return len(data)


def _graph_counts(path: Path) -> tuple[int, int] | tuple[None, None]:
    if not path.is_file():
        return None, None
    root = ET.parse(path).getroot()
    nodes = 0
    edges = 0
    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name == "node":
            nodes += 1
        elif local_name == "edge":
            edges += 1
    return nodes, edges


def collect_cache_stats(cache_dir: Path) -> dict[str, Any]:
    graph_nodes, graph_edges = _graph_counts(
        cache_dir / "graph_chunk_entity_relation.graphml"
    )
    total_bytes = (
        sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
        if cache_dir.is_dir()
        else 0
    )
    return {
        "cache_dir": cache_dir,
        "exists": cache_dir.is_dir(),
        "total_bytes": total_bytes,
        "full_documents": _json_length(cache_dir / "kv_store_full_docs.json"),
        "text_chunks": _json_length(cache_dir / "kv_store_text_chunks.json"),
        "community_reports": _json_length(
            cache_dir / "kv_store_community_reports.json"
        ),
        "llm_cache_entries": _json_length(
            cache_dir / "kv_store_llm_response_cache.json"
        ),
        "chunk_vectors": _json_length(cache_dir / "vdb_chunks.json", "data"),
        "entity_vectors": _json_length(cache_dir / "vdb_entities.json", "data"),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
    }


def format_stats(stats: dict[str, Any]) -> str:
    labels = [
        ("完整文档", "full_documents"),
        ("文本分块", "text_chunks"),
        ("社区报告", "community_reports"),
        ("LLM 缓存条目", "llm_cache_entries"),
        ("Chunk 向量", "chunk_vectors"),
        ("实体向量", "entity_vectors"),
        ("图谱节点", "graph_nodes"),
        ("图谱边", "graph_edges"),
    ]
    lines = [f"缓存目录: {stats['cache_dir']}", f"缓存存在: {stats['exists']}"]
    for label, key in labels:
        value = stats[key]
        lines.append(f"  {label}: {'未生成' if value is None else value}")
    lines.append(f"  占用空间: {stats['total_bytes'] / 1024 / 1024:.2f} MiB")
    return "\n".join(lines)


def run_check() -> int:
    from nano_graphrag._config import get_settings
    from nano_graphrag._embedding import get_bge_m3_model

    settings = get_settings()
    print(format_settings_summary(settings))

    import numpy as np
    import torch

    cuda_available = torch.cuda.is_available()
    print(f"torch: {torch.__version__}")
    print(f"CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif settings.embedding_device.startswith("cuda"):
        raise RuntimeError(
            "EMBEDDING_DEVICE 配置为 CUDA，但当前 torch 无法使用 CUDA"
        )

    chapters = split_book_into_chapters(read_book())
    print(f"书籍: {BOOK_PATH}")
    print(f"章节数: {len(chapters)}")

    print("正在加载本地 BGE-M3 并编码测试文本...")
    model = get_bge_m3_model()
    vector = np.asarray(
        model.encode(
            ["《三国演义》预检"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    if vector.shape != (1, 1024):
        raise RuntimeError(f"BGE-M3 输出维度异常: {vector.shape}")
    norm = float(np.linalg.norm(vector[0]))
    if not np.isfinite(vector).all() or abs(norm - 1.0) > 1e-4:
        raise RuntimeError(f"BGE-M3 输出未正确归一化，范数={norm}")
    print(f"Embedding: shape={vector.shape}, dtype={vector.dtype}, norm={norm:.6f}")
    print("环境检查通过；此命令未调用 DeepSeek。")
    return 0


def run_check_api() -> int:
    from nano_graphrag._config import get_settings
    from nano_graphrag._llm import deepseek_cheap_complete

    settings = get_settings()
    print(format_settings_summary(settings))
    response = asyncio.run(
        deepseek_cheap_complete(
            "只回复 OK，不要解释。",
            max_tokens=8,
        )
    )
    if not response or not response.strip():
        raise RuntimeError("DeepSeek API 返回了空响应")
    print(f"DeepSeek API 检查通过，模型响应: {response.strip()}")
    return 0


def run_build(scope: str, confirmed: bool) -> int:
    cache_dir = CACHE_DIRS[scope]
    print(validate_build_target(scope, cache_dir, confirmed))
    documents = documents_for_scope(scope)
    print(f"准备构建 {scope} 数据：{len(documents)} 个章节文档")
    print(
        "注意：每个 chunk 至少执行两次实体抽取 LLM 调用，"
        "随后还会生成实体摘要和社区报告。"
    )

    started_at = time.perf_counter()
    insert_documents(cache_dir, documents)
    elapsed = time.perf_counter() - started_at
    print(f"构建完成，耗时 {elapsed:.2f} 秒")
    print(format_stats(collect_cache_stats(cache_dir)))
    return 0


def run_query(
    scope: str,
    mode: str,
    question: str | None,
    top_k: int,
    only_need_context: bool,
) -> int:
    from nano_graphrag.base import QueryParam
    from nano_graphrag.graphrag import GraphRAG

    cache_dir = CACHE_DIRS[scope]
    modes = ["local", "naive", "global"] if mode == "all" else [mode]
    for query_mode in modes:
        require_cache_for_mode(cache_dir, query_mode)

    graph = GraphRAG(**graphrag_options(cache_dir))
    for query_mode in modes:
        query = question or DEFAULT_QUESTIONS[query_mode]
        print(f"\n[{query_mode}] {query}")
        if query_mode == "global":
            print("global 查询会执行社区摘要 map-reduce，调用量通常最高。")
        started_at = time.perf_counter()
        result = graph.query(
            query,
            param=QueryParam(
                mode=query_mode,
                top_k=top_k,
                only_need_context=only_need_context,
            ),
        )
        elapsed = time.perf_counter() - started_at
        print(f"耗时 {elapsed:.2f} 秒")
        print(result)
    return 0


def run_stats(scope: str) -> int:
    cache_dir = CACHE_DIRS[scope]
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"缓存目录不存在: {cache_dir}")
    print(format_stats(collect_cache_stats(cache_dir)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek + 本地 BGE-M3 构建和查询《三国演义》。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="检查配置、CUDA、BGE-M3 和书籍格式")
    subparsers.add_parser("check-api", help="执行一次短 DeepSeek API 检查")

    build_parser_ = subparsers.add_parser("build", help="建立 Embedding 和图谱缓存")
    build_parser_.add_argument("--scope", choices=["sample", "full"], required=True)
    build_parser_.add_argument(
        "--yes",
        action="store_true",
        help="确认执行高消耗的整书构建（仅 full 需要）",
    )

    query_parser = subparsers.add_parser("query", help="查询已建立的缓存")
    query_parser.add_argument("--scope", choices=["sample", "full"], required=True)
    query_parser.add_argument(
        "--mode",
        choices=["local", "naive", "global", "all"],
        default="local",
    )
    query_parser.add_argument("--question", help="覆盖该模式的默认测试问题")
    query_parser.add_argument("--top-k", type=int, default=20)
    query_parser.add_argument(
        "--context",
        action="store_true",
        help="只检查检索上下文",
    )

    stats_parser = subparsers.add_parser("stats", help="查看缓存规模和图谱统计")
    stats_parser.add_argument("--scope", choices=["sample", "full"], required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_output()
    from nano_graphrag.prompt_zh import install_chinese_prompts

    install_chinese_prompts()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            return run_check()
        if args.command == "check-api":
            return run_check_api()
        if args.command == "build":
            return run_build(args.scope, args.yes)
        if args.command == "query":
            return run_query(
                args.scope,
                args.mode,
                args.question,
                args.top_k,
                args.context,
            )
        if args.command == "stats":
            return run_stats(args.scope)
    except KeyboardInterrupt:
        print("\n操作已取消。", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"未处理的命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
