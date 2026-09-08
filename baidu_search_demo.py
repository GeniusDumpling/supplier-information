import functools
import json
import logging
import os
import re
import time
from datetime import datetime

import requests
import yaml

logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, base_delay: float = 1.0, backoff: float = 2.0,
               exceptions=(requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
    """指数退避重试装饰器。达到 max_retries 后重抛最后一次异常。"""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_retries:
                        raise
                    wait = base_delay * (backoff ** (attempt - 1))
                    logger.warning(
                        f"{fn.__name__} 失败（{e}），{wait}s 后重试 {attempt + 1}/{max_retries} ...")
                    time.sleep(wait)
            raise RuntimeError(f"{fn.__name__} 重试耗尽")

        return wrapper

    return deco


# 全局运行 ID：由 pipeline.py 设置，用于统一产物命名；未设置时回退当前时间戳
RUN_ID = None


def set_run_id(rid: str) -> None:
    global RUN_ID
    RUN_ID = rid


def stamp() -> str:
    """返回产物命名时间戳：优先使用 run_id，否则取当前时间。"""
    return RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")


INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_results", "_index")


def load_known_urls() -> set:
    """读取跨 run 已知 URL 累积库（用于增量：跳过已抓取过的 URL）。"""
    p = os.path.join(INDEX_DIR, "known_urls.txt")
    if not os.path.exists(p):
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def record_known_urls(urls) -> int:
    """把成功抓取的 URL 追加进已知库，返回新增条数。"""
    p = os.path.join(INDEX_DIR, "known_urls.txt")
    os.makedirs(INDEX_DIR, exist_ok=True)
    known = load_known_urls()
    new = 0
    with open(p, "a", encoding="utf-8") as f:
        for u in urls:
            if u and u not in known:
                f.write(u + "\n")
                known.add(u)
                new += 1
    return new

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None


API_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"


def base_dir() -> str:
    """脚本所在目录。"""
    return os.path.dirname(os.path.abspath(__file__))


def load_conf(config_name: str = "conf.yaml") -> dict:
    """从同级 conf.yaml 读取全部配置（search / llm / verify 等）。"""
    path = os.path.join(base_dir(), config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_api_key() -> str:
    """从环境变量或 .env 读取 API Key（环境变量优先，适合服务器定时调度）。"""
    env_val = os.environ.get("BAIDU_QIANFAN_API_KEY", "").strip()
    if env_val:
        return env_val.strip('"').strip("'")
    # .env 文件与脚本同级，作为本地开发兜底
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "BAIDU_QIANFAN_API_KEY":
                        return value.strip().strip('"').strip("'")
    return ""


def sanitize_filename(name: str) -> str:
    """去掉文件名中的非法字符，避免 Windows 下保存报错。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "search_result"


def fetch_fulltext(url: str, config=None, retries: int = 3, delay: float = 1.0) -> str:
    """用 trafilatura 抓取网页完整正文，带指数重试。失败返回空字符串。"""
    if trafilatura is None:
        return ""
    for attempt in range(1, retries + 1):
        try:
            kwargs = {"config": config} if config is not None else {}
            downloaded = trafilatura.fetch_url(url, **kwargs)
            if downloaded:
                text = trafilatura.extract(downloaded, include_links=False)
                if text:
                    return text
        except Exception as e:
            logger.warning(f"抓取失败 {url[:50]}...（{e}），第 {attempt}/{retries} 次尝试")
        if attempt < retries and delay > 0:
            time.sleep(delay)
    return ""


@with_retry()
def search(query: str, conf: dict) -> dict:
    """调用百度千帆 AI 联网搜索接口，返回解析后的响应 JSON。"""
    payload = {
        "messages": [{"role": "user", "content": query}],
    }

    # 可选搜索参数（按 conf.yaml 配置填充）
    if conf.get("edition"):
        payload["edition"] = conf["edition"]
    if conf.get("search_source"):
        payload["search_source"] = conf["search_source"]
    if conf.get("instruction"):
        payload["instruction"] = conf["instruction"]
    recency = conf.get("recency_filter")
    if recency and str(recency).lower() not in ("none", "null"):
        payload["search_recency_filter"] = recency

    resource_types = conf.get("resource_types")
    if resource_types:
        payload["resource_type_filter"] = [
            {"type": rt.get("type"), "top_k": rt.get("top_k", 0)} for rt in resource_types
        ]

    if conf.get("domain_filter"):
        payload["search_filter"] = {"match": {"site": conf["domain_filter"]}}

    if conf.get("safe_search"):
        payload["safe_search"] = True

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }

    timeout = conf.get("timeout", 30)
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=timeout)
    response.raise_for_status()
    return response.json()


def save_snapshot(query: str, refs: list, out_dir: str) -> str:
    """保存快照：原始 JSON + 可读 Markdown 文本快照。返回 Markdown 快照路径。"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{sanitize_filename(query)}_{stamp()}")

    # 快照 1：原始 JSON
    json_path = base + "_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"references": refs}, f, ensure_ascii=False, indent=2)

    # 快照 2：可读 Markdown
    md_path = base + "_snapshot.md"
    lines = [f"# 搜索结果快照：{query}", f"\n搜索时间：{datetime.now():%Y-%m-%d %H:%M:%S}", f"结果数量：{len(refs)}\n"]
    for i, ref in enumerate(refs, 1):
        lines.append(f"## {i}. {ref.get('title', '无标题')}")
        if ref.get("url"):
            lines.append(f"地址：{ref['url']}")
        if ref.get("source"):
            lines.append(f"来源：{ref['source']}")
        if ref.get("date"):
            lines.append(f"日期：{ref['date']}")
        content = (ref.get("content") or "无内容摘要").replace("\u0004", "...").replace("\u0005", "...")
        lines.append(f"\n{content}\n")
        lines.append("---")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path


def save_urls(query: str, refs: list, out_dir: str) -> str:
    """把所有网页 URL 保存到单独文档 txt。返回路径。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sanitize_filename(query)}_urls.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"关键词：{query}\n")
        f.write(f"导出时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"URL 数量：{len(refs)}\n\n")
        for i, ref in enumerate(refs, 1):
            src = f" [{ref.get('source', '')}]" if ref.get("source") else ""
            f.write(f"{i}.{src} {ref.get('title', '无标题')}\n{ref.get('url', '')}\n\n")
    return path


def save_fulltext_snapshot(query: str, refs: list, out_dir: str, delay: float = 1.0,
                           fetch_retries: int = 3, record_new: bool = True) -> str:
    """抓取每个 URL 完整正文并流式写入 Markdown，支持断点续抓与失败 URL 记录。

    - 同 run（stamp() 相同、文件已存在）重跑时，跳过已成功抓取的 URL；
    - 每条抓取成功后立即 flush 写入，中断后不丢失已完成部分；
    - 抓取失败的 URL 另存到 *_failed_urls.txt 便于补抓。
    需要 trafilatura。
    """
    if trafilatura is None:
        raise RuntimeError("未安装 trafilatura，请先执行：pip install trafilatura")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{sanitize_filename(query)}_{stamp()}")
    path = base + "_fulltext.md"
    fail_path = base + "_failed_urls.txt"

    total = len(refs)

    # 断点：若文件已存在（同 run 重跑），收集已成功 URL
    done_urls = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("地址："):
                    done_urls.add(line[len("地址："):].strip())

    ok = skipped = 0
    ok_urls = []
    mode = "a" if os.path.exists(path) else "w"
    if mode == "w":
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 搜索结果全文快照：{query}\n\n"
                    f"抓取时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n"
                    f"结果数量：{total}\n\n")

    failed = []
    with open(path, mode, encoding="utf-8") as f:
        for i, ref in enumerate(refs, 1):
            url = ref.get("url", "")
            title = ref.get("title", "无标题")
            f.write(f"## {i}. {title}\n地址：{url}\n")
            if ref.get("source"):
                f.write(f"来源：{ref['source']}\n")
            if ref.get("date"):
                f.write(f"日期：{ref['date']}\n")
            f.write("\n")

            if not url:
                f.write("（无 URL，跳过）\n\n---\n")
                continue
            if url in done_urls:
                skipped += 1
                logger.info(f"[{i}/{total}] 已抓取，跳过：{url[:50]}...")
                f.write("\n---\n")
                continue

            logger.info(f"[{i}/{total}] 抓取全文：{url[:60]}...")
            text = fetch_fulltext(url, retries=fetch_retries)
            if text:
                ok += 1
                f.write(text)
                done_urls.add(url)
                ok_urls.append(url)
            else:
                logger.warning(f"[{i}/{total}] 抓取失败或页面无正文：{url[:60]}...")
                f.write("（抓取失败或页面无正文）")
                if url:
                    failed.append(url)
            f.write("\n\n---\n")
            f.flush()
            if delay > 0:
                time.sleep(delay)

    if failed:
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        logger.warning(f"抓取失败 {len(failed)} 条，失败列表：{fail_path}")

    if record_new and ok_urls:
        recorded = record_known_urls(ok_urls)
        logger.info(f"已记录 {recorded} 个新 URL 到已知库（known_urls.txt）。")

    logger.info(f"全文抓取成功 {ok}/{total}（跳过 {skipped}）。")
    return path


def main() -> str:
    """百度搜索→合并快照→可选抓取全文。返回主产物（fulltext 优先）路径。"""
    conf = load_conf()
    search_conf = conf.get("search", {})

    query = str(search_conf.get("keyword", "")).strip()
    if not query:
        logger.warning("未配置搜索关键词，请在 conf.yaml 的 search.keyword 中填写。")
        return ""

    search_results_dir = os.path.join(base_dir(), "search_results")
    os.makedirs(search_results_dir, exist_ok=True)
    # 每次运行的结果统一放入 厂商名_日期时间戳 子目录
    out_dir = os.path.join(search_results_dir, f"{sanitize_filename(query)}_{stamp()}")
    logger.info(f"正在搜索：{query} ...")
    logger.info(f"输出目录：{out_dir}")

    refs = []
    use_baidu = search_conf.get("use_baidu", True)
    if not use_baidu:
        logger.warning("use_baidu=false，当前流程仅支持百度搜索，无法继续。")
        return ""
    # 百度千帆 AI 搜索
    if not get_api_key():
        logger.warning("未找到 BAIDU_QIANFAN_API_KEY，请在 .env 或环境变量中配置。")
        return ""
    logger.info("[百度] 调用千帆 AI 搜索 ...")
    baidu_result = search(query, search_conf)
    refs = baidu_result.get("references", [])
    for _r in refs:
        _r["source"] = "baidu"
    logger.info(f"[百度] 获取 {len(refs)} 条结果。")

    logger.info(f"合并去重后共 {len(refs)} 条结果。")
    if not refs:
        logger.warning("没有获取到任何结果，请检查关键词或 API 配置。")
        return ""

    md_path = save_snapshot(query, refs, out_dir)
    url_path = save_urls(query, refs, out_dir)

    logger.info(f"搜索结果快照已保存：{md_path}")
    logger.info(f"网页 URL 文档已保存：{url_path}")

    result_path = md_path
    if search_conf.get("fetch_fulltext"):
        # 增量模式：只抓取新增 URL，跳过已知 URL（省资源、聚焦新内容）
        fetch_refs = refs
        if search_conf.get("incremental", False):
            known = load_known_urls()
            new_refs = [r for r in refs if r.get("url") and r["url"] not in known]
            logger.info(f"增量模式：共 {len(refs)} 条，新增 {len(new_refs)} 条，跳过已知 "
                        f"{len(refs) - len(new_refs)} 条。")
            fetch_refs = new_refs
            if not fetch_refs:
                logger.info("无新增 URL，抓取阶段跳过。")
                return result_path

        logger.info("开始抓取网页完整正文 ...")
        full_path = save_fulltext_snapshot(
            query, fetch_refs, out_dir,
            delay=search_conf.get("fetch_delay", 1.0),
            fetch_retries=int(search_conf.get("fetch_retries", 3)),
        )
        logger.info(f"网页全文快照已保存：{full_path}")
        result_path = full_path
    return result_path


if __name__ == "__main__":
    main()