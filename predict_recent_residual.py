"""CLI for the causal last-week + recent-4h-residual predictor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from load15_recent_residual_predictor import forecast_from_project, save_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", required=True, help="Prediction origin; naive values are interpreted as Asia/Shanghai")
    parser.add_argument("--config", default="my_load15_config.json")
    parser.add_argument("--model-config", default="model_checkpoints/recent_residual_v1/model.json")
    parser.add_argument("--beta", type=float, help="Explicit override; normally omit to use the frozen model config")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = forecast_from_project(args.origin, args.config, beta=args.beta, model_config_path=args.model_config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output_dir) if args.output_dir else Path("my_outputs") / f"recent_residual_inference_{stamp}"
    save_forecast(result, output)
    print(f"output_dir={output.resolve()}")
    print(f"origin={result.metadata['origin']}")
    print(f"mode={result.metadata['mode']}")
    print(f"correction_kw={result.metadata['applied_correction_kw']}")
    print(f"forecast_shape={result.forecast.shape}")


if __name__ == "__main__":
    main()
