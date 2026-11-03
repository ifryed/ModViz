from __future__ import annotations

import argparse
import sys

from .api import visualize
from .model_loader import load_model, parse_json_object
from .tracer import make_inputs, parse_shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torch-flow-viz",
        description="Create an interactive HTML data-flow diagram from a PyTorch model file.",
    )
    parser.add_argument("file", help="Python file containing the model")
    parser.add_argument("--model", help="Module symbol: an nn.Module instance, class, or factory")
    parser.add_argument("--input-shape", action="append", required=True, metavar="DIMS", help="Example input shape, e.g. 1,3,224,224. Repeat for multiple inputs.")
    parser.add_argument("--init-args", default="{}", metavar="JSON", help="Keyword arguments passed to the model constructor/factory")
    parser.add_argument("--dtype", default="float32", help="Input torch dtype (default: float32)")
    parser.add_argument(
        "--input-dtype",
        action="append",
        metavar="DTYPE",
        help="Per-input dtype; repeat in the same order as --input-shape (useful for GNN edge indices)",
    )
    parser.add_argument("--device", default="cpu", help="Tracing device (default: cpu)")
    parser.add_argument("-o", "--output", default="network_flow.html", help="Output HTML file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        kwargs = parse_json_object(args.init_args, "--init-args")
        model = load_model(args.file, args.model, kwargs)
        model.to(args.device)
        shapes = [parse_shape(value) for value in args.input_shape]
        dtypes = args.input_dtype if args.input_dtype else args.dtype
        output = visualize(model, make_inputs(shapes, dtypes, args.device), args.output)
    except Exception as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

