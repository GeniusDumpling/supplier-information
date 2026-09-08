"""供应商定向验证（独立流程，可定时执行）。

读取 pipeline 维护的"明确供应关系"文档（search_results/_index/confirmed_relations.md），
对其中未验证（或全部）的供应商用百度定向搜索"{供应商} 大疆 供应商"，抓取前几条全文后
交给 LLM 判定该供应商与大疆的真实供应关系，输出验证报告，并把验证状态回写该文档。

运行：python supplier_verify.py
"""
import logging
import os
import re
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
    load_confirmed_relations,
    save_confirmed_relations,
)

logger = logging.getLogger(__name__)


def get_deepseek_key() -> str:
    """从环境变量或 .env 读取 DeepSeek API Key（环境变量优先）。"""
    env_val = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_val:
        return env_val.strip('"').strip("'")
    env_path = os.path.join(base_dir(), ".env")
    key = ""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == "DEEPSEEK_API_KEY":
                        key = v.strip().strip('"').strip("'")
    return key


VERIFY_SYSTEM_PROMPT = (
    "你是一名供应链验证分析师。以下是根据定向搜索『某公司 与 大疆』关系获取的若干网页全文。"
    "请基于这些证据，判定该公司与大疆之间是否存在真实的供货/代工/代理/销售等实质性关系，"
    "并给出供应内容。\n"
    "判定标准（结论三选一）：\n"
    "1. 确认：原文直接写明该公司向大疆供应某产品/服务/代工/零部件，或双方确为供应链合作关系。\n"
    "2. 否定：原文明确该公司并非大疆供应商，或两者无实质供应关系。\n"
    "3. 待确认：只有线索但证据不足，无法下结论。\n"
    "另给出可信度：明确 / 疑似 / 不相关。\n"
    '请严格按如下 Markdown 格式输出，不要输出多余解释：\n\n'
    '- 验证结论：确认 / 否定 / 待确认\n'
    '- 可信度：明确 / 疑似 / 不相关\n'
    '- 供应内容/模块：\n'
    '- 主要证据（引用原文，每条约一行，附来源URL）：\n'
)


def build_user_content(supplier: str, supply_hint: str, docs: list) -> str:
    lines = [
        f"待验证供应商：{supplier}",
        f"原始抽取的供应内容：{supply_hint or '（无）'}",
        "",
    ]
    if not docs:
        lines.append("（未获取到该供应商与大疆相关的网页全文，请基于'无证据'给出结论）")
        return "\n".join(lines)
    for i, d in enumerate(docs, 1):
        lines.append(f"===== 网页 {i} =====")
        lines.append(f"标题：{d['title']}")
        lines.append(f"地址：{d['url']}")
        lines.append(d["body"])
        lines.append("")
    return "\n".join(lines)


def call_llm(api_key: str, conf: dict, user_content: str) -> str:
    url = conf["api_base"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": conf["model"],
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": conf.get("temperature", 0.2),
        "max_tokens": conf.get("max_tokens", 4096),
        "stream": False,
    }
    max_retries = conf.get("max_retries", 3)
    timeout = conf.get("timeout", 120)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt >= max_retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"    LLM 请求失败（{e}），{wait}s 后重试 {attempt + 1}/{max_retries} ...")
            time.sleep(wait)
    raise RuntimeError("LLM 请求失败")


def verify_supplier(api_key, sup, search_conf, llm_conf, verify_conf):
    """对单个供应商做定向搜索 + 抓取 + LLM 判定。"""
    query = f"{sup['supplier']} 大疆 供应商"
    search_limit = int(verify_conf.get("search_limit", 20))
    fetch_limit = int(verify_conf.get("fetch_limit", 5))
    fetch_delay = float(verify_conf.get("fetch_delay", 0.5))

    # 1) 百度定向搜索
    try:
        result = search(query, search_conf)
        refs = result.get("references", []) or []
    except Exception as e:
        return {
            "supplier": sup["supplier"], "query": query,
            "searched": 0, "fetched": 0,
            "verdict": f"搜索失败：{e}", "confidence": "", "supply": "", "evidence": [],
        }
    refs = refs[:search_limit]
    logger.info(f"    [搜索] {sup['supplier']} -> 取前 {len(refs)} 条")

    # 2) 抓取前 fetch_limit 条全文
    docs = []
    for ref in refs[:fetch_limit]:
        url = ref.get("url", "")
        title = ref.get("title", "无标题")
        if not url:
            continue
        text = fetch_fulltext(url)
        if text:
            docs.append({"title": title, "url": url, "body": text})
        if fetch_delay > 0:
            time.sleep(fetch_delay)

    # 3) LLM 判定
    logger.info(f"    [LLM] 判定 {sup['supplier']}（抓取 {len(docs)} 条）...")
    user_content = build_user_content(sup["supplier"], sup.get("supply", ""), docs)
    verdict_md = call_llm(api_key, llm_conf, user_content)

    # 4) 解析结论中的关键字段用于汇总表
    m = re.search(r"^-\s*验证结论：\s*(.*)$", verdict_md, flags=re.M)
    verdict = m.group(1).strip() if m else "待确认"
    m = re.search(r"^-\s*可信度：\s*(.*)$", verdict_md, flags=re.M)
    confidence = m.group(1).strip() if m else ""
    m = re.search(r"^-\s*供应内容/模块：\s*(.*)$", verdict_md, flags=re.M)
    supply = m.group(1).strip() if m else ""

    return {
        "supplier": sup["supplier"],
        "query": query,
        "searched": len(refs),
        "fetched": len(docs),
        "verdict": verdict,
        "confidence": confidence,
        "supply": supply,
        "evidence": verdict_md,
        "orig_sources": sup.get("sources", []),
        "orig_supply": sup.get("supply", ""),
    }


def main(confirmed_path: str = "") -> str:
    """独立验证流程：读"明确供应关系"文档，按模式验证并回写状态。

    confirmed_path 非空时使用该文件，否则默认 search_results/_index/confirmed_relations.md。
    mode=unverified（默认）只验证未验证项；mode=all 验证全部。
    """
    path = os.path.abspath(confirmed_path) if confirmed_path else CONFIRMED_RELATIONS_MD
    if not os.path.exists(path):
        raise SystemExit(f"未找到明确供应关系文档：{path}（请先运行 pipeline.py 分析生成）。")
    logger.info(f"读取明确供应关系文档：{path}")

    conf = load_conf()
    search_conf = conf.get("search", {})
    llm_conf = conf.get("llm", {})
    verify_conf = conf.get("verify", {}) or {}
    if not verify_conf.get("enabled", True):
        logger.warning("verify.enabled=false，跳过验证。")
        return ""

    api_key = get_deepseek_key()
    if not api_key:
        raise SystemExit("未配置 DEEPSEEK_API_KEY，请在 .env 或环境变量中填写。")
    if not get_api_key():
        raise SystemExit("未配置 BAIDU_QIANFAN_API_KEY，请在 .env 或环境变量中填写。")

    sec = load_confirmed_relations(path)
    if not sec:
        logger.warning("明确供应关系文档为空，无待验证项。")
        return ""

    mode = str(verify_conf.get("mode", "unverified")).strip().lower()
    targets = []
    for name, rec in sec.items():
        if mode == "all" or rec.get("verify_status", "未验证") != "已验证":
            targets.append((name, {
                "supplier": name,
                "supply": rec.get("supply", ""),
                "sources": rec.get("sources", []),
            }))

    max_n = int(verify_conf.get("max_suppliers", 0) or 0)
    if max_n > 0:
        targets = targets[:max_n]
    total = len(targets)
    logger.info(f"待验证供应商（mode={mode}）：{total}/{len(sec)} 家")

    results = []
    for idx, (name, sup) in enumerate(targets, 1):
        logger.info(f"[{idx}/{total}] 验证：{name}")
        try:
            res = verify_supplier(api_key, sup, search_conf, llm_conf, verify_conf)
        except Exception as e:
            logger.warning(f"    {name} 验证失败：{e}")
            res = {"supplier": name, "query": f"{name} 大疆 供应商",
                   "searched": 0, "fetched": 0, "verdict": f"验证失败：{e}",
                   "confidence": "", "supply": "", "evidence": "",
                   "orig_sources": sup.get("sources", []),
                   "orig_supply": sup.get("supply", "")}
        results.append(res)
        # 回写验证状态（"失败" 关键字标记验证失败，否则视为已验证）
        sec[name]["verify_status"] = "验证失败" if "失败" in res["verdict"] else "已验证"
        sec[name]["verify_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sec[name]["verify_verdict"] = f"{res['verdict']}；{res['supply']}"

    # 回写文档
    save_confirmed_relations(sec, path)

    # 生成验证报告（与来源文档同目录，独立时间戳）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(path), f"verification_{ts}.md")
    lines = [
        f"# 供应商验证报告（{ts}）",
        f"\n依据文档：{os.path.basename(path)}",
        f"验证时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"验证模式：{mode}，待验证供应商数：{total}",
        "",
        "## 验证结果汇总",
        "",
        "| # | 供应商 | 原始供应内容 | 验证结论 | 可信度 | 证据 |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        evidence = f"搜索{r['searched']}/抓取{r['fetched']}条"
        lines.append(f"| {i} | {r['supplier']} | {(r['orig_supply'] or '—')[:40].replace('|', '/')} "
                     f"| {r['verdict']} | {r['confidence'] or '—'} | {evidence} |")
    lines.append("")

    lines.append("## 详细证据")
    for i, r in enumerate(results, 1):
        lines.append(f"\n### {i}. {r['supplier']}")
        lines.append(f"- 验证关键词：{r['query']}")
        lines.append(f"- 原始供应内容：{r['orig_supply'] or '（无）'}")
        if r.get("orig_sources"):
            lines.append(f"- 原始来源：{'；'.join(r['orig_sources'])}")
        if not r.get("evidence"):
            lines.append(f"\n{r['verdict']}")
            continue
        lines.append("")
        lines.append(r["evidence"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"验证完成，结果已保存：{out_path}")
    return out_path


if __name__ == "__main__":
    # 独立运行时显式配置日志（无上层 handler 也有输出）
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    main()