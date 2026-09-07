import json
import os
import re
import sys
from datetime import datetime
import yaml
import requests

def base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_yaml(name: str) -> dict:
    path = os.path.join(base_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_deepseek_key() -> str:
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
    return key or os.environ.get("DEEPSEEK_API_KEY", "")


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
    resp = requests.post(url, headers=headers, json=payload,
                         timeout=conf.get("timeout", 120))
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    # 定位最新 *_fulltext.md
    out_dir = os.path.join(base_dir(), "search_results")
    md_files = [f for f in os.listdir(out_dir) if f.endswith("_fulltext.md")] if os.path.exists(out_dir) else []
    if not md_files:
        raise SystemExit("未找到 *_fulltext.md，请先运行 baidu_search_demo.py 生成全文快照。")
    md_files.sort(reverse=True)
    md_path = os.path.join(out_dir, md_files[0])

    conf = load_yaml("conf.yaml")
    llm_conf = conf.get("llm", {})

    api_key = get_deepseek_key()
    if not api_key:
        raise SystemExit("未配置 DEEPSEEK_API_KEY，请在 .env 中填写（DEEPSEEK_API_KEY=你的Key）。")

    items = parse_fulltext(md_path)
    total = len(items)
    print(f"解析到 {total} 条网页全文：{md_path}")
    if total == 0:
        raise SystemExit("没有可分析的正文。")

    chunk_size = int(llm_conf.get("chunk_size", 10))
    chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]

    results = []
    for n, chunk in enumerate(chunks, 1):
        indices = f"{((n - 1) * chunk_size + 1)}-{((n - 1) * chunk_size + len(chunk))}"
        print(f"[分片 {n}/{len(chunks)}] 调用 {llm_conf.get('model')} 分析 {len(chunk)} 条 ...")
        try:
            summary = call_llm(api_key, llm_conf, build_user_content(chunk))
        except Exception as e:
            err = f"（第 {n} 片分析失败：{e}）"
            print("  " + err)
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

    print(f"分析完成，结果已保存：{out_path}")


if __name__ == "__main__":
    main()