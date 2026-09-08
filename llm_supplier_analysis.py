import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

import yaml

import requests

logger = logging.getLogger(__name__)

def base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_yaml(name: str) -> dict:
    path = os.path.join(base_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def parse_fulltext(path: str) -> list:
    """解析 *_fulltext.md，返回 [{title, url, date, body}, ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 按 "## 序号. " 切分成段
    parts = re.split(r"^## \d+\. ", text, flags=re.M)
    if len(parts) <= 1:
        # 备用切分：找不到 ## 就整体当作一段
        return [{"title": "全文", "url": "", "date": "", "body": text.strip()}]

    items = []
    for part in parts[1:]:
        lines = part.split("\n")
        title = lines[0].strip()
        url = date = ""
        bodies = []
        for line in lines[1:]:
            if line.startswith("地址："):
                url = line[len("地址："):].strip()
            elif line.startswith("日期："):
                date = line[len("日期："):].strip()
            elif line.startswith("来源：") or line.strip() == "---":
                continue
            else:
                bodies.append(line)
        body = "\n".join(bodies).strip()
        if body and ("（抓取失败" not in body) and ("（无 URL" not in body):
            items.append({"title": title, "url": url, "date": date, "body": body})
    return items


SYSTEM_PROMPT = (
    "你是一名供应链分析师。根据给定的网页全文，识别其中可能存在的供应关系"
    "（供应商与采购方之间的供货关系，例如某公司向大疆/某整机厂供应某模块、零件或服务）。\n"
    "判定标准（分三档）：\n"
    "1. 明确：原文直接写明采购方与供应商及供货内容（如\"X是Y的供应商\"、\"Y向X采购Z\"）。\n"
    "2. 疑似/待验证：原文隐含有供货关系线索但未明说，例如\"X是Y领域的供应商\"、\"X已供货/配套某类产品\"而采购方需要推断、或列举的候选供应商但未点名大疆为采购方。这类要显式标为\"疑似\"，并在依据里说明推断点。\n"
    "3. 不相关：与供应关系无关的，不要列出。\n"
    '请严格按如下 Markdown 格式输出，不要输出多余解释，每条关系一行：\n\n'
    '### 关系N\n'
    '- 采购方：\n'
    '- 供应商：\n'
    '- 供应内容/模块：\n'
    '- 可信度：明确 / 疑似\n'
    '- 依据（引用原文）：\n'
    '- 来源（网页原始URL）：\n'
)


def build_user_content(chunk: list) -> str:
    lines = []
    for i, item in enumerate(chunk, 1):
        lines.append(f"===== 网页 {i} =====")
        lines.append(f"标题：{item['title']}")
        if item['url']:
            lines.append(f"地址：{item['url']}")
        if item['date']:
            lines.append(f"日期：{item['date']}")
        lines.append(item['body'])
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.HTTPError) as e:
            if attempt >= max_retries:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM 请求失败（{e}），{wait}s 后重试 {attempt + 1}/{max_retries} ...")
            time.sleep(wait)
    raise RuntimeError("LLM 请求失败")


def parse_relations(summary_path: str) -> list:
    """解析 summary 中所有关系块，返回 [{supplier, supply, credibility, buyer, url}, ...]。"""
    with open(summary_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"^###\s*关系\d+\s*$", text, flags=re.M)

    def field(block, name):
        m = re.search(rf"^-\s*{name}：\s*(.*)$", block, flags=re.M)
        return m.group(1).strip() if m else ""

    rels = []
    for block in blocks:
        if "- 供应商：" not in block or "- 可信度：" not in block:
            continue
        supplier = field(block, "供应商")
        if not supplier:
            continue
        rels.append({
            "supplier": supplier,
            "supply": field(block, "供应内容/模块"),
            "credibility": field(block, "可信度"),
            "buyer": field(block, "采购方"),
            "url": (field(block, "来源（网页原始URL）").split("；")[0].split(";")[0].strip()),
        })
    return rels


REL_INDEX_JSON = os.path.join(base_dir(), "search_results", "_index", "supplier_relations.json")


def update_relation_index(summary_path: str, run_id: str) -> str:
    """把本次关系累积到 _index/supplier_relations.json，并生成差分报告。

    返回增量报告路径。new=相对历史首次出现的供应商；not_mentioned=本次未再被提及的已知供应商。
    """
    rels = parse_relations(summary_path)

    idx = {}
    if os.path.exists(REL_INDEX_JSON):
        try:
            with open(REL_INDEX_JSON, "r", encoding="utf-8") as f:
                idx = json.load(f)
        except Exception:
            idx = {}
    cumulative = idx.get("cumulative", {})
    prev = set(cumulative.keys())  # 本次 merge 前已存在的供应商

    today_suppliers = set()
    for r in rels:
        s = r["supplier"]
        today_suppliers.add(s)
        rec = cumulative.get(s)
        if rec is None:
            cumulative[s] = {
                "supplier": s, "first_run": run_id, "last_run": run_id,
                "count": 1, "credibility": r["credibility"],
                "sources": [r["url"]] if r["url"] else [],
            }
        else:
            rec["last_run"] = run_id
            rec["count"] += 1
            rec["credibility"] = r["credibility"]
            if r["url"] and r["url"] not in rec["sources"]:
                rec["sources"].append(r["url"])

    new_suppliers = [r for r in rels if r["supplier"] not in prev]
    not_mentioned = sorted(s for s in prev if s not in today_suppliers)

    idx["updated_run"] = run_id
    idx["cumulative"] = cumulative
    with open(REL_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    ts = run_id.rsplit("_", 1)[-1] if "_" in run_id else run_id
    delta_path = os.path.join(os.path.dirname(REL_INDEX_JSON), f"incremental_{ts}.md")
    lines = [
        f"# 供应商关系增量报告（run={run_id}）",
        f"\n分析时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"本次解析到 {len(rels)} 条关系，涉及 {len(today_suppliers)} 个供应商\n",
        "## 新增供应商（相对历史首次出现）",
    ]
    if new_suppliers:
        for r in new_suppliers:
            lines.append(f"- **{r['supplier']}**｜{r['supply'] or '—'}｜可信度:{r['credibility']}"
                         + (f"｜{r['url']}" if r["url"] else ""))
    else:
        lines.append("- （无）")

    lines.append("\n## 本次出现的历史已知供应商")
    if today_suppliers:
        for s in sorted(today_suppliers):
            rec = cumulative[s]
            lines.append(f"- {s}｜可信度:{rec['credibility']}｜累计 {rec['count']} 次｜末次 {rec['last_run']}")
    else:
        lines.append("- （无）")

    lines.append("\n## 本次未再提及的已知供应商（待观察：可能退链/此处无新资料）")
    if not_mentioned:
        for s in not_mentioned:
            rec = cumulative[s]
            lines.append(f"- {s}｜累计 {rec['count']} 次｜末次 {rec['last_run']}")
    else:
        lines.append("- （无）")

    with open(delta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return delta_path


CONFIRMED_RELATIONS_MD = os.path.join(
    base_dir(), "search_results", "_index", "confirmed_relations.md")


def load_confirmed_relations(path: str = CONFIRMED_RELATIONS_MD) -> dict:
    """解析 confirmed_relations.md ->
    {supplier: {supply,first,last,count,sources,verify_status,verify_time,verify_verdict}}。"""
    sec = {}
    if not os.path.exists(path):
        return sec
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    keymap = {"供应内容": "supply", "首次出现": "first", "末次出现": "last",
              "出现次数": "count", "验证状态": "verify_status",
              "验证时间": "verify_time", "验证结论": "verify_verdict"}
    for part in re.split(r"^##\s+", text, flags=re.M)[1:]:
        lines = part.split("\n")
        name = lines[0].strip()
        if not name:
            continue
        rec = {"sources": [], "verify_status": "未验证", "verify_time": "", "verify_verdict": ""}
        collecting = False
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("- 来源URL："):
                collecting = True
                continue
            if collecting:
                if re.match(r"^- https?://", line):
                    rec["sources"].append(line[2:].strip())
                    continue
                collecting = False
            m = re.match(r"^- ([^：]+)：\s*(.*)$", line)
            if m and m.group(1) in keymap:
                rec[keymap[m.group(1)]] = m.group(2).strip()
        sec[name] = rec
    return sec


def save_confirmed_relations(sec: dict, path: str = CONFIRMED_RELATIONS_MD) -> str:
    """把明确供应关系写回 confirmed_relations.md。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# 明确供应关系（可信度=明确）",
        "<!-- 由 llm_supplier_analysis 每次分析后增量维护；supplier_verify 独立定时读取并回写验证状态 -->",
        "",
    ]
    for name in sorted(sec.keys()):
        rec = sec[name]
        lines.append(f"## {name}")
        lines.append(f"- 供应内容：{rec.get('supply', '')}")
        lines.append(f"- 首次出现：{rec.get('first', '')}")
        lines.append(f"- 末次出现：{rec.get('last', '')}")
        lines.append(f"- 出现次数：{rec.get('count', '')}")
        lines.append("- 来源URL：")
        for u in rec.get("sources", []):
            lines.append(f"    - {u}")
        lines.append(f"- 验证状态：{rec.get('verify_status', '未验证')}")
        lines.append(f"- 验证时间：{rec.get('verify_time', '')}")
        lines.append(f"- 验证结论：{rec.get('verify_verdict', '')}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def update_confirmed_relations(summary_path: str, run_id: str) -> str:
    """把本次 summary 中"可信度=明确"的关系增量累积到 confirmed_relations.md。"""
    confirmed = [r for r in parse_relations(summary_path) if r["credibility"] == "明确"]
    sec = load_confirmed_relations()
    for r in confirmed:
        s = r["supplier"]
        rec = sec.get(s)
        if rec is None:
            sec[s] = {
                "supply": r["supply"], "first": run_id, "last": run_id, "count": 1,
                "sources": [r["url"]] if r["url"] else [],
                "verify_status": "未验证", "verify_time": "", "verify_verdict": "",
            }
        else:
            rec["last"] = run_id
            rec["count"] = int(rec.get("count") or 0) + 1
            if r["url"] and r["url"] not in rec["sources"]:
                rec["sources"].append(r["url"])
    path = save_confirmed_relations(sec)
    logger.info(f"明确供应关系已更新：{path}（当前 {len(sec)} 家）")
    return path


def main(fulltext_path: str = "") -> str:
    """对全文快照执行 LLM 供应关系分析。返回总结文件路径。

    fulltext_path 非空时使用该文件，否则定位最新 *_fulltext.md。
    """
    # 定位 *_fulltext.md，输出目录与其同目录（即  厂商名_时间戳 子目录）
    search_results = os.path.join(base_dir(), "search_results")
    if fulltext_path:
        if not fulltext_path.endswith("_fulltext.md"):
            raise SystemExit(
                f"传入的不是 *_fulltext.md（{fulltext_path}），"
                "拒绝把搜索快照/其他文档当作全文进行分析。")
        md_path = fulltext_path
    else:
        files = glob.glob(os.path.join(search_results, "**", "*_fulltext.md"), recursive=True) \
            if os.path.exists(search_results) else []
        if not files:
            raise SystemExit("未找到 *_fulltext.md，请先运行搜索脚本生成全文快照。")
        files.sort(key=os.path.getmtime, reverse=True)
        md_path = files[0]
    out_dir = os.path.dirname(md_path)

    conf = load_yaml("conf.yaml")
    llm_conf = conf.get("llm", {})

    api_key = get_deepseek_key()
    if not api_key:
        raise SystemExit("未配置 DEEPSEEK_API_KEY，请在 .env 或环境变量中填写（DEEPSEEK_API_KEY=你的Key）。")

    items = parse_fulltext(md_path)
    total = len(items)
    logger.info(f"解析到 {total} 条网页全文：{md_path}")
    if total == 0:
        raise SystemExit("没有可分析的正文。")

    chunk_size = int(llm_conf.get("chunk_size", 10))
    chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]

    results = []
    for n, chunk in enumerate(chunks, 1):
        indices = f"{((n - 1) * chunk_size + 1)}-{((n - 1) * chunk_size + len(chunk))}"
        logger.info(f"[分片 {n}/{len(chunks)}] 调用 {llm_conf.get('model')} 分析 {len(chunk)} 条 ...")
        try:
            summary = call_llm(api_key, llm_conf, build_user_content(chunk))
        except Exception as e:
            err = f"（第 {n} 片分析失败：{e}）"
            logger.warning("  " + err)
            results.append(err)
            continue
        results.append(f"\n# 分片 {n}（网页 {indices}）")
        results.append(summary)

    # 保存结果
    keyword = os.path.basename(md_path)
    # 取 md 文件名里的关键词段（去掉 _time_fulltext）
    m = re.search(r"^(.*?)_\d{8}_\d{6}_fulltext\.md$", keyword)
    keyword = m.group(1) if m else os.path.splitext(keyword)[0].replace("_fulltext", "")
    out_path = os.path.join(out_dir, f"{keyword}_supplier_summary.md")

    header = [
        f"# 供应关系分析总结：{keyword}",
        f"\n来源：{os.path.basename(md_path)}",
        f"分析模型：{llm_conf.get('model')}",
        f"分析时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n" + "\n".join(results))

    logger.info(f"分析完成，结果已保存：{out_path}")
    # 增量：累积全量关系并生成差分报告
    run_id = os.path.basename(out_dir)
    try:
        delta_path = update_relation_index(out_path, run_id)
        logger.info(f"增量报告已生成：{delta_path}")
    except Exception as e:
        logger.warning(f"生成增量报告失败：{e}")
    # 维护"明确供应关系"文档，供独立验证流程定时消费
    try:
        conf_path = update_confirmed_relations(out_path, run_id)
        logger.info(f"明确供应关系已维护：{conf_path}")
    except Exception as e:
        logger.warning(f"维护明确供应关系失败：{e}")
    return out_path


if __name__ == "__main__":
    main()