import os
import sys
import copy
import argparse
from typing import Any, Dict, List, Tuple, Optional

import yaml
import pandas as pd
import scanpy as sc

sys.path.append(os.path.abspath("./src"))

from dmt import DyMoTree
from utils.metrics import calculate_fate_metrics


SWEEP_STRING_KEYS = {"emb_key", "mode", "pre_train","adata_path"}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_allowed_sweep_list(key: str, value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        raise ValueError(f"参数 `{key}` 不能是空 list。")

    if key in SWEEP_STRING_KEYS:
        if not all(isinstance(v, str) for v in value):
            raise ValueError(f"`{key}` 若为 list，则必须全是字符串。")
        return True

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return True

    return False


def find_sweep_candidates(obj: Any, path: str = "") -> List[Tuple[str, List[Any]]]:
    candidates = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k

            if isinstance(v, dict):
                candidates.extend(find_sweep_candidates(v, current_path))
            elif isinstance(v, list):
                # seed 单独处理，不算普通 sweep 参数
                if current_path.endswith("seed"):
                    continue
                if is_allowed_sweep_list(k, v):
                    candidates.append((current_path, v))

    return candidates


def get_seed_list(cfg: Dict[str, Any]) -> List[int]:
    seed = cfg["model"]["seed"]

    if isinstance(seed, list):
        if len(seed) == 0:
            raise ValueError("seed list 不能为空。")
        if not all(isinstance(s, int) for s in seed):
            raise ValueError("seed list 必须全是 int。")
        return seed

    if isinstance(seed, int):
        return [seed]

    raise ValueError("seed 必须是 int 或 list[int]。")


def get_by_path(cfg: Dict[str, Any], path: str) -> Any:
    cur = cfg
    for part in path.split("."):
        cur = cur[part]
    return cur


def set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def validate_terminals(terminals: List[str]) -> None:
    if not isinstance(terminals, list) or len(terminals) != 2:
        raise ValueError("`model.terminal` 必须是长度为 2 的列表。")
    if not all(isinstance(x, str) for x in terminals):
        raise ValueError("`model.terminal` 必须是字符串列表。")


def compute_fate_bias(hspc, terminals: List[str], fate_bias_col: str) -> None:
    validate_terminals(terminals)

    first_terminal = terminals[0]
    second_terminal = terminals[1]

    first_col = f"{first_terminal}_fate"
    second_col = f"{second_terminal}_fate"

    if first_col not in hspc.obs:
        raise KeyError(f"未找到列: {first_col}")
    if second_col not in hspc.obs:
        raise KeyError(f"未找到列: {second_col}")

    numerator = hspc.obs[first_col]
    denominator = hspc.obs[first_col] + hspc.obs[second_col]
    hspc.obs[fate_bias_col] = numerator / denominator


def build_dmt(cfg: Dict[str, Any]) -> DyMoTree:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    adata = sc.read_h5ad(data_cfg["adata_path"])

    return DyMoTree(
        adata=adata,
        k=model_cfg["k"],
        progenitor=model_cfg["progenitor"],
        terminal=model_cfg["terminal"],
        lineage_col=model_cfg["lineage_col"],
        emb_key=model_cfg["emb_key"],
        device=model_cfg["device"],
        seed=model_cfg["seed"],
    )


def run_one_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    validate_terminals(cfg["model"]["terminal"])

    dmt = build_dmt(cfg)

    dmt.lineage_graph(
        mask_threshold=cfg["lineage_graph"]["mask_threshold"],
        epsilon=cfg["lineage_graph"]["epsilon"],
        mode=cfg["lineage_graph"]["mode"],
    )

    dmt.train(
        pre_train=cfg["train"]["pre_train"],
        lr=cfg["train"]["lr"],
        iter=cfg["train"]["iter"],
        sample_ratio=cfg["train"]["sample_ratio"],
        alpha=cfg["train"]["alpha"],
    )

    progenitor = cfg["model"]["progenitor"]
    terminals = cfg["model"]["terminal"]
    fate_bias_col = cfg["evaluation"]["fate_bias_col"]
    ground_truth_col = cfg["evaluation"]["ground_truth_col"]
    threshold = cfg["evaluation"]["threshold"]

    hspc = dmt.treedata.get_node(progenitor, adata_object=True)
    compute_fate_bias(hspc, terminals, fate_bias_col)

    if ground_truth_col not in hspc.obs:
        raise KeyError(f"未找到 ground truth 列: {ground_truth_col}")
    if progenitor == 'Undifferentiated':
        truth_fate = 1-hspc.obs[ground_truth_col].values
    else:
        truth_fate = hspc.obs[ground_truth_col].values
    predict_fate = hspc.obs[fate_bias_col].values

    metrics = calculate_fate_metrics(
        truth_fate,
        predict_fate,
        threshold=threshold,
    )

    return metrics


def prepare_runs(base_cfg: Dict[str, Any]):
    candidates = find_sweep_candidates(base_cfg)
    seed_list = get_seed_list(base_cfg)

    if len(candidates) > 1:
        names = ", ".join(path for path, _ in candidates)
        raise ValueError(
            f"检测到多个普通 list 参数：{names}。当前只允许一个普通参数为 list，seed 可单独为 list。"
        )

    runs = []

    if len(candidates) == 0:
        for seed in seed_list:
            cfg_i = copy.deepcopy(base_cfg)
            cfg_i["model"]["seed"] = seed
            runs.append((None, None, seed, cfg_i))
        return None, runs

    sweep_path, sweep_values = candidates[0]

    for value in sweep_values:
        for seed in seed_list:
            cfg_i = copy.deepcopy(base_cfg)
            set_by_path(cfg_i, sweep_path, value)
            cfg_i["model"]["seed"] = seed
            runs.append((sweep_path, value, seed, cfg_i))

    return sweep_path, runs


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat


def build_result_row(
    run_index: int,
    sweep_param: Optional[str],
    sweep_value: Any,
    seed: int,
    cfg: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "run_index": run_index,
        "seed": seed,
        "sweep_param": sweep_param,
        "sweep_value": sweep_value,
        "adata_path": cfg["data"]["adata_path"],
        "progenitor": cfg["model"]["progenitor"],
        "terminal_1": cfg["model"]["terminal"][0],
        "terminal_2": cfg["model"]["terminal"][1],
        "k": cfg["model"]["k"],
        "emb_key": cfg["model"]["emb_key"],
        "device": cfg["model"]["device"],
        "lineage_col": cfg["model"]["lineage_col"],
        "mask_threshold": cfg["lineage_graph"]["mask_threshold"],
        "epsilon": cfg["lineage_graph"]["epsilon"],
        "mode": cfg["lineage_graph"]["mode"],
        "pre_train": cfg["train"]["pre_train"],
        "lr_formal": cfg["train"]["lr"]["formal"],
        "lr_intra": cfg["train"]["lr"]["intra"],
        "lr_lineage": cfg["train"]["lr"]["lineage"],
        "iter_formal": cfg["train"]["iter"]["formal"],
        "iter_intra": cfg["train"]["iter"]["intra"],
        "iter_lineage": cfg["train"]["iter"]["lineage"],
        "sample_ratio": cfg["train"]["sample_ratio"],
        "alpha": cfg["train"]["alpha"],
        "ground_truth_col": cfg["evaluation"]["ground_truth_col"],
        "fate_bias_col": cfg["evaluation"]["fate_bias_col"],
        "threshold": cfg["evaluation"]["threshold"],
    }

    row.update(flatten_metrics(metrics))
    return row


def save_results_csv(rows: List[Dict[str, Any]], output_csv: str) -> None:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fate prediction with config-based sweep and save results to CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/fate_prediction_results.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, runs = prepare_runs(cfg)

    result_rows = []

    for idx, (sweep_param, sweep_value, seed, run_cfg) in enumerate(runs, start=1):
        print("=" * 60)
        if sweep_param is None:
            print(f"Run {idx}: seed = {seed}")
        else:
            print(f"Run {idx}: {sweep_param} = {sweep_value}, seed = {seed}")

        metrics = run_one_experiment(run_cfg)
        print("Metrics:")
        print(metrics)

        row = build_result_row(
            run_index=idx,
            sweep_param=sweep_param,
            sweep_value=sweep_value,
            seed=seed,
            cfg=run_cfg,
            metrics=metrics,
        )
        result_rows.append(row)

    save_results_csv(result_rows, args.output_csv)
    print("=" * 60)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()