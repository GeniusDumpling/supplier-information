"""统一调度入口：串起 搜索→抓取全文→LLM 供应关系分析→供应商验证。

- 生成统一 run_id 贯穿各阶段产物命名（run_id 由各模块 stamp() 使用）；
- 日志写入 logs/pipeline_<run_id>.log（RotatingFileHandler，含控制台输出）；
- 运行结束写入状态摘要 logs/pipeline_<run_id>.json。

运行：python pipeline.py
"""
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime

from baidu_search_demo import base_dir, load_conf, set_run_id, main as search_main
from llm_supplier_analysis import main as llm_main
from supplier_verify import main as verify_main

logger = logging.getLogger("pipeline")


def setup_logging(run_id: str) -> str:
    """配置 root logger：控制台 + 轮转日志文件。返回日志文件路径。"""
    log_dir = os.path.join(base_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"pipeline_{run_id}.log")

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
    ]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        for h in handlers:
            h.setFormatter(fmt)
            root.addHandler(h)
    return log_path


def try_stage(name: str, fn, *args):
    """执行单个阶段，捕获异常并记录日志。失败返回 None。"""
    try:
        return fn(*args)
    except SystemExit as e:
        logger.error(f"[{name}] 中止：{e}")
    except Exception as e:
        logger.error(f"[{name}] 异常：{e}")
    return None


def main() -> int:
    conf = load_conf()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(run_id)
    set_run_id(run_id)

    logger.info(f"==== pipeline 启动 run_id={run_id} ====")
    logger.info(f"日志文件：{log_path}")

    artifacts = {"run_id": run_id}
    failures = []

    # 阶段 1：搜索（百度+Brave）+ 抓取全文
    logger.info("[阶段 1/3] 搜索并抓取全文 ...")
    fulltext = try_stage("阶段 1/3", search_main)
    if not fulltext:
        failures.append("阶段 1：搜索/抓取未产出 fulltext")
        logger.error("[阶段 1/3] 未产出全文快照，中止后续阶段。")
    else:
        artifacts["fulltext"] = fulltext
        logger.info(f"[阶段 1/3] 完成 -> {fulltext}")

        # 阶段 2：LLM 供应关系分析
        logger.info("[阶段 2/3] LLM 供应关系分析 ...")
        summary = try_stage("阶段 2/3", llm_main, fulltext)
        if not summary:
            failures.append("阶段 2：LLM 供应关系分析未产出 summary")
            logger.error("[阶段 2/3] 未产出总结文件。")
        else:
            artifacts["summary"] = summary
            logger.info(f"[阶段 2/3] 完成 -> {summary}")

            # 阶段 3：供应商验证
            logger.info("[阶段 3/3] 供应商定向验证 ...")
            verification = try_stage("阶段 3/3", verify_main, summary)
            if verification:
                artifacts["verification"] = verification
                logger.info(f"[阶段 3/3] 完成 -> {verification}")
            else:
                logger.warning("[阶段 3/3] 已跳过（verify.enabled=false 或没有可验证供应商）。")

    status = "failed" if failures else "ok"
    summary_payload = {
        "status": status,
        "run_id": run_id,
        "artifacts": artifacts,
        "failures": failures,
    }
    meta_path = os.path.join(base_dir(), "logs", f"pipeline_{run_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    logger.info(f"==== pipeline 结束 status={status} ====")
    for k, v in artifacts.items():
        logger.info(f"  {k}: {v}")
    for e in failures:
        logger.error(f"  FAIL: {e}")
    logger.info(f"状态摘要：{meta_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())