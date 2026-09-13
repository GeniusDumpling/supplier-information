"""供应商零件/模块详情查询脚本。

针对"明确供应关系"文档（confirmed_relations.md）中的某个已验证供应商，用百度搜索该厂商
供应大疆的零件/模块的具体技术信息（型号、性能、参数、适配机型等），抓取返回网页全文后
交给 LLM 提炼为结构化报告，输出 *_part_detail.md。

用法：
    python supplier_part_detail.py [厂商名]
不带参数时默认使用"中润光学"。
"""
import logging
import os
import re
import sys
import time
from datetime import datetime

import requests

from baidu_search_demo import (
    base_dir,
    load_conf,
    get_api_key,
    search,
    fetch_fulltext,
)
from llm_supplier_analysis import (
    CONFIRMED_RELATIONS_MD,
    get_deepseek_key,
    load_confirmed_relations,
)

logger = logging.getLogger(__name__)


PART_SYSTEM_PROMPT = (
    "你是一名供应链零部件分析师。以下是定向搜索某厂商（大疆供应商）获取的若干网页全文。"
    "请基于这些证据，提炼该厂商供应给大疆的零件/模块的具体技术信息。\n"
    "请重点给出（可引用原文）：\n"
    "1. 供应零件/模块名称\n"
    "2. 具体型号/系列\n"
    "3. 性能与参数（如分辨率、焦距、倍率、尺寸、重量、接口、材料等）\n"
    "4. 适配的大疆机型/平台\n"
    "5. 供应关系依据（引用原文 + 来源URL）\n"
    "判定要求：只输出有网页证据支撑的信息；查不到型号/参数时明确写\"未检索到\""
    "；不要编造。整体输出一段 Markdown，含标题和分节、条目列表。"
)


PART_SLICE_PROMPT = (
    "你是一名供应链零部件分析师。以下是一批网页全文（定向搜索某大疆供应商的零件/模块信息）。"
    "请提炼这批网页中与该厂商相关的零件/模块具体技术信息（名称、型号、性能参数、适配机型、来源URL）。\n"
    "只输出有网页证据支撑的信息，无证据则写\"未检索到\"，不要编造。用 Markdown 条目输出。"
)

PART_MERGE_PROMPT = (
    "你是一名供应链零部件分析师。以下是对同一大疆供应商多批网页分别提炼的结果。"
    "请合并去重这些批次结果，输出该厂商供应大疆零件/模块的最终结构化详情，包含：\n"
    "1. 供应零件/模块名称\n2. 具体型号/系列\n3. 性能与参数\n4. 适配大疆机型/平台\n5. 供应关系依据（含来源URL）\n"
    "合并去重并保留证据，查不到的项写\"未检索到\"。整体一段 Markdown。"
)


def build_merge_content(supplier: str, slice_results: list) -> str:
    lines = [f"待合并厂商：{supplier}", ""]
    for i, r in enumerate(slice_results, 1):
        lines.append(f"== 批次 {i} ==")
        lines.append(r)
        lines.append("")
    return "\n".join(lines)


def build_user_content(supplier: str, supply_hint: str, docs: list) -> str:
    lines = [
        f"待查厂商：{supplier}",
        f"已知供应内容（供参考）：{supply_hint or '（无）'}",
        "",
    ]
    if not docs:
        lines.append("（未获取到与该厂商相关的网页全文，请基于'无证据'说明）")
        return "\n".join(lines)
    for i, d in enumerate(docs, 1):
        lines.append(f"===== 网页 {i} =====")
        lines.append(f"标题：{d['title']}")
        lines.append(f"地址：{d['url']}")
        lines.append(d["body"])
        lines.append("")
    return "\n".join(lines)


def call_llm(api_key: str, conf: dict, content: str,
             sys_prompt: str = PART_SYSTEM_PROMPT) -> str:
    url = conf["api_base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": conf["model"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": conf.get("temperature", 0.2),
        "max_tokens": conf.get("max_tokens", 4096),
        "stream": False,
    }
    max_retries = int(conf.get("max_retries", 3))
    timeout = conf.get("timeout", 120)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.HTTPError) as e:
            if attempt >= max_retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM 请求失败（{e}），{wait}s 后重试 {attempt + 1}/{max_retries} ...")
            time.sleep(wait)
    raise RuntimeError("LLM 请求失败")


def collect_part_info(supplier: str, supply_hint: str, search_conf, llm_conf, pd_conf, api_key):
    """百度搜索→抓取全部结果→分片提炼→合并。返回 {keyword, found, urls, result_md}。"""
    # 构造搜索关键词：厂商 + 大疆 + 供应内容关键词 + 型号参数
    supply_phasis = ""
    if supply_hint:
        # 截取供应内容中较长的一段作为搜索补充词
        supply_phasis = supply_hint.split("（")[0].strip()
    keyword = " ".join(x for x in [supplier, "大疆", supply_phasis, "型号 参数 规格"] if x)
    logger.info(f"定向搜索：{keyword}")

    try:
        result = search(keyword, search_conf)
        refs = result.get("references", []) or []
    except Exception as e:
        return {"keyword": keyword, "found": 0, "urls": [], "result_md": f"搜索失败：{e}"}

    # 覆盖单次请求返回的全部结果（默认 50，可用 part_detail.fetch_top 调整）
    fetch_top = int(pd_conf.get("fetch_top", 50))
    chunk_size = int(pd_conf.get("chunk_size", 10))
    fetch_delay = float(search_conf.get("fetch_delay", 0.5))

    docs, urls = [], []
    for ref in refs[:fetch_top]:
        url = ref.get("url", "")
        title = ref.get("title", "无标题")
        if not url:
            continue
        urls.append(url)
        text = fetch_fulltext(url)
        if text:
            docs.append({"title": title, "url": url, "body": text})
        if fetch_delay > 0:
            time.sleep(fetch_delay)
    logger.info(f"搜索返回 {len(refs)} 条，抓取 {len(docs)}/{len(urls)} 条全文")

    if not docs:
        return {"keyword": keyword, "found": len(refs), "urls": urls,
                "result_md": "（未抓取到可用网页全文）"}

    # 分片提炼
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    logger.info(f"LLM 分片提炼：{len(chunks)} 片（每片 {chunk_size} 条）...")
    slice_results = []
    for n, chunk in enumerate(chunks, 1):
        logger.info(f"  [片 {n}/{len(chunks)}] 提炼 {len(chunk)} 条 ...")
        slice_results.append(call_llm(
            api_key, llm_conf,
            build_user_content(supplier, supply_hint, chunk),
            PART_SLICE_PROMPT))

    if len(slice_results) == 1:
        result_md = slice_results[0]
    else:
        logger.info("LLM 合并各批次提炼结果 ...")
        result_md = call_llm(api_key, llm_conf,
                             build_merge_content(supplier, slice_results),
                             PART_MERGE_PROMPT)
    return {"keyword": keyword, "found": len(refs), "urls": urls, "result_md": result_md}


def main() -> int:
    supplier = (sys.argv[1].strip() if len(sys.argv) > 1 else "").strip() or "中润光学"

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")

    conf = load_conf()
    search_conf = conf.get("search", {})
    llm_conf = conf.get("llm", {})
    pd_conf = conf.get("part_detail", {}) or {}
    api_key = get_deepseek_key()
    if not api_key:
        raise SystemExit("未配置 DEEPSEEK_API_KEY，请在 .env 或环境变量中填写。")
    if not get_api_key():
        raise SystemExit("未配置 BAIDU_QIANFAN_API_KEY，请在 .env 或环境变量中填写。")

    # 从 confirmed 文档取该供应商的供应内容（供关键词），找不到则置空
    sec = load_confirmed_relations()
    supply_hint = (sec.get(supplier, {}).get("supply", "") or "") if sec else ""

    info = collect_part_info(supplier, supply_hint, search_conf, llm_conf, pd_conf, api_key)

    out_dir = os.path.join(base_dir(), "search_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{supplier}_part_detail_{ts}.md")

    lines = [
        f"# {supplier} 供应大疆零件/模块详情",
        f"\n查询厂商：{supplier}",
        f"已知供应内容：{supply_hint or '（无）'}",
        f"搜索关键词：{info['keyword']}",
        f"查询时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"百度返回：{info['found']} 条；抓取来源：{len(info['urls'])} 个",
        "",
        "## 提炼结果",
        "",
        info["result_md"],
        "",
        "## 检索到的来源 URL",
        "",
    ]
    lines.extend(f"- {u}" for u in info["urls"])
    if not info["urls"]:
        lines.append("（无来源）")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"查询完成，结果已保存：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())