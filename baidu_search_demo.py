import requests
import json
import os
import re
import time
from datetime import datetime

import yaml

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
    """从同级 conf.yaml 读取全部配置（search / brave 等）。"""
    path = os.path.join(base_dir(), config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_api_key() -> str:
    """从 .env 或环境变量读取 API Key。"""
    # .env 文件与脚本同级
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "BAIDU_QIANFAN_API_KEY":
                        return value.strip().strip('"').strip("'")
    return os.environ.get("BAIDU_QIANFAN_API_KEY", "")


def get_brave_key() -> str:
    """从 .env 或环境变量读取 Brave Search API Key。"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    if key.strip() == "BRAVE_SEARCH_API_KEY":
                        return value.strip().strip('"').strip("'")
    return os.environ.get("BRAVE_SEARCH_API_KEY", "")


def sanitize_filename(name: str) -> str:
    """去掉文件名中的非法字符，避免 Windows 下保存报错。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "search_result"


def fetch_fulltext(url: str, config=None) -> str:
    """用 trafilatura 抓取网页完整正文。失败返回空字符串。"""
    if trafilatura is None:
        return ""
    try:
        kwargs = {"config": config} if config is not None else {}
        downloaded = trafilatura.fetch_url(url, **kwargs)
        if not downloaded:
            return ""
        return trafilatura.extract(downloaded, include_links=False) or ""
    except Exception:
        return ""


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


BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
# 百度 recency_filter 到 Brave freshness 的映射
_RECENCY_TO_FRESHNESS = {
    "week": "pw", "month": "pm", "semiyear": "py", "year": "py",
}


def brave_search(query: str, search_conf: dict, brave_conf: dict) -> list:
    """调用 Brave Search API，返回 [{title, url, date, content}, ...]（按 web 结果）。"""
    params = {
        "q": query,
        "count": int(brave_conf.get("count", 20)),
        "country": brave_conf.get("country", "CN"),
        "search_lang": brave_conf.get("search_lang", "zh"),
        "safesearch": brave_conf.get("safesearch", "moderate"),
    }
    # 显式 freshness 优先；否则按百度 recency_filter 自动映射
    freshness = brave_conf.get("freshness", "") or ""
    if not freshness:
        recency = str(search_conf.get("recency_filter", "")).lower()
        if recency in _RECENCY_TO_FRESHNESS:
            freshness = _RECENCY_TO_FRESHNESS[recency]
    if freshness:
        params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": get_brave_key(),
    }
    resp = requests.get(BRAVE_SEARCH_URL, params=params, headers=headers,
                        timeout=brave_conf.get("timeout", 30))
    resp.raise_for_status()
    data = resp.json()

    refs = []
    for item in data.get("web", {}).get("results", []):
        refs.append({
            "title": item.get("title", "无标题"),
            "url": item.get("url", ""),
            "date": item.get("age", ""),
            "content": item.get("description", ""),
            "source": "brave",
        })
    return refs


def merge_refs(*ref_lists: list) -> list:
    """多来源结果合并去重（按 URL）。返回合并后的引用列表。"""
    seen = set()
    merged = []
    for refs in ref_lists:
        for ref in refs:
            url = ref.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(ref)
    return merged


def save_snapshot(query: str, refs: list, out_dir: str) -> str:
    """保存快照：原始 JSON + 可读 Markdown 文本快照。返回 Markdown 快照路径。"""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{sanitize_filename(query)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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


def save_fulltext_snapshot(query: str, refs: list, out_dir: str, delay: float = 1.0) -> str:
    """抓取每个 URL 的完整正文，保存为带全文的 Markdown 快照。需要 trafilatura。"""
    if trafilatura is None:
        raise RuntimeError("未安装 trafilatura，请先执行：pip install trafilatura")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{sanitize_filename(query)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path = base + "_fulltext.md"

    lines = [f"# 搜索结果全文快照：{query}", f"\n抓取时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
             f"结果数量：{len(refs)}\n"]

    ok = 0
    for i, ref in enumerate(refs, 1):
        url = ref.get("url", "")
        title = ref.get("title", "无标题")
        lines.append(f"## {i}. {title}")
        lines.append(f"地址：{url}")
        if ref.get("source"):
            lines.append(f"来源：{ref['source']}")
        if ref.get("date"):
            lines.append(f"日期：{ref['date']}")
        lines.append("")

        if not url:
            lines.append("（无 URL，跳过）")
            lines.append("")
            lines.append("---")
            continue

        print(f"[{i}/{len(refs)}] 抓取全文：{url[:60]}...")
        text = fetch_fulltext(url)
        if text:
            ok += 1
            lines.append(text)
        else:
            lines.append("（抓取失败或页面无正文）")
        lines.append("")
        lines.append("---")

        if delay > 0:
            time.sleep(delay)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"全文抓取成功 {ok}/{len(refs)} 条。")
    return path


def main():
    conf = load_conf()
    search_conf = conf.get("search", {})
    brave_conf = conf.get("brave", {}) or {}

    query = str(search_conf.get("keyword", "")).strip()
    if not query:
        print("未配置搜索关键词，请在 conf.yaml 的 search.keyword 中填写。")
        return

    out_dir = os.path.join(base_dir(), "search_results")
    print(f"正在搜索：{query} ...")

    ref_lists = []

    # 1) 百度千帆 AI 搜索
    use_baidu = search_conf.get("use_baidu", True)
    if use_baidu:
        if not get_api_key():
            print("未找到 BAIDU_QIANFAN_API_KEY，请在 .env 中配置（BAIDU_QIANFAN_API_KEY=你的Key）。")
            return
        print(f"[百度] 调用千帆 AI 搜索 ...")
        baidu_result = search(query, search_conf)
        refs_baidu = baidu_result.get("references", [])
        for _r in refs_baidu:
            _r["source"] = "baidu"
        print(f"[百度] 获取 {len(refs_baidu)} 条结果。")
        ref_lists.append(refs_baidu)
    else:
        refs_baidu = []

    # 2) Brave Search（独立英文关键词）
    refs_brave = []
    if brave_conf.get("enabled", True):
        if not get_brave_key():
            print("未找到 BRAVE_SEARCH_API_KEY，请在 .env 中配置（BRAVE_SEARCH_API_KEY=你的Key）。")
            return
        brave_query = str(brave_conf.get("keyword_en", "")).strip() or query
        print(f"[Brave] 调用 Brave Search API（关键词：{brave_query}）...")
        try:
            refs_brave = brave_search(brave_query, search_conf, brave_conf)
            print(f"[Brave] 获取 {len(refs_brave)} 条结果。")
            ref_lists.append(refs_brave)
        except Exception as e:
            print(f"[Brave] 搜索失败：{e}")

    # 合并去重
    refs = merge_refs(*ref_lists)
    print(f"合并去重后共 {len(refs)} 条结果。")
    if not refs:
        print("没有获取到任何结果，请检查关键词或 API 配置。")
        return

    md_path = save_snapshot(query, refs, out_dir)
    url_path = save_urls(query, refs, out_dir)

    print(f"搜索结果快照已保存：{md_path}")
    print(f"网页 URL 文档已保存：{url_path}")

    if search_conf.get("fetch_fulltext"):
        print("开始抓取网页完整正文 ...")
        full_path = save_fulltext_snapshot(
            query, refs, out_dir,
            delay=search_conf.get("fetch_delay", 1.0),
        )
        print(f"网页全文快照已保存：{full_path}")


if __name__ == "__main__":
    main()