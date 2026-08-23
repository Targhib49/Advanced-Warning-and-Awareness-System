"""Export the trained eye-state model for the Pi.

You train on arm64 macOS and run on arm64 Linux. That is a real porting step,
and the latency budget in master plan §7.2 depends on getting it right:

    torch checkpoint -> ONNX (fp32) -> ONNX (int8, static)

Static quantization (not dynamic) because this is a convolutional model —
dynamic quantization only helps MatMul/Gemm layers and would leave the convs,
which are all of the compute, untouched.

Every stage is verified numerically against the stage before it. An export that
silently changes the model's predictions is worse than no export.

    python -m training.export
    python -m training.export --checkpoint artifacts/eyestate.pt --bench 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from .config import OUT_DIR, TrainConfig
from .model import EyeStateNet

OPSET = 18


def _require_onnx():
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ONNX export needs onnx and onnxruntime:\n"
            "    pip install onnx onnxruntime\n"
            f"({exc})"
        )


def load_model(checkpoint: Path) -> tuple[EyeStateNet, int]:
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}.\n"
            f"Train one first:  python -m training.train_eyestate --smoke"
        )
    blob = torch.load(checkpoint, map_location="cpu")
    crop_px = blob.get("crop_px", TrainConfig().crop_px)
    model = EyeStateNet(in_px=crop_px)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, crop_px


def export_fp32(model: EyeStateNet, crop_px: int, path: Path) -> None:
    dummy = torch.randn(1, 1, crop_px, crop_px)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["eye"], output_names=["logits"],
        # Batch stays dynamic so the runtime can process both eyes in one call.
        dynamic_axes={"eye": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )


class _CalibReader:
    """Feeds representative inputs to the static quantizer.

    Calibration data must LOOK LIKE what the model sees in the cabin. Random
    noise here would set the activation ranges from a distribution the model
    never encounters and quietly cost accuracy. Pass real eye crops when you
    have them; this falls back to the synthetic set only so the pipeline is
    runnable before the dataset lands.
    """

    def __init__(self, crop_px: int, n: int = 128, real_dir: Optional[Path] = None):
        from onnxruntime.quantization import CalibrationDataReader  # noqa: F401
        self._it: Iterator = iter(self._batches(crop_px, n, real_dir))

    @staticmethod
    def _batches(crop_px: int, n: int, real_dir: Optional[Path]):
        if real_dir is not None and real_dir.exists():
            from PIL import Image
            paths = sorted(
                p for p in real_dir.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )[:n]
            for p in paths:
                img = Image.open(p).convert("L").resize((crop_px, crop_px))
                arr = (np.asarray(img, dtype=np.float32) / 255.0)[None, None]
                yield {"eye": arr}
            return
        from .data import SyntheticEyeDataset
        ds = SyntheticEyeDataset(n=n, crop_px=crop_px, seed=99)
        for i in range(len(ds)):
            x, _ = ds[i]
            yield {"eye": x.unsqueeze(0).numpy()}

    def get_next(self):
        return next(self._it, None)


def _pre_process(src: Path, dst: Path) -> Path:
    """Shape-infer before quantizing, degrading rather than failing.

    torch's dynamo exporter emits a dynamic batch axis that onnxruntime's
    symbolic shape inference cannot always resolve. Skipping inference costs a
    little quantization quality; failing the export costs the whole int8 path,
    so we walk down instead of falling over. The same fallback matters on the
    Pi, where the onnxruntime build differs again.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    for skip in (False, True):
        try:
            quant_pre_process(str(src), str(dst), skip_symbolic_shape=skip)
            return dst
        except Exception as exc:
            if skip:
                print(f"  shape inference unavailable ({type(exc).__name__}); "
                      f"quantizing the unprocessed graph")
                return src
    return src


def quantize_int8(src: Path, dst: Path, crop_px: int, calib_dir: Optional[Path]) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    prepped = _pre_process(src, src.with_name(src.stem + "_prep.onnx"))
    try:
        quantize_static(
            model_input=str(prepped),
            model_output=str(dst),
            calibration_data_reader=_CalibReader(crop_px, real_dir=calib_dir),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
    finally:
        if prepped != src:
            prepped.unlink(missing_ok=True)


def _session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # match a realistic per-stage Pi budget
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def verify(model: EyeStateNet, path: Path, crop_px: int, n: int = 64) -> tuple[float, float]:
    """Compare ONNX against torch on the same inputs.

    Returns (max logit difference, agreement rate on P(closed) > 0.5).
    """
    rng = np.random.default_rng(7)
    x = rng.random((n, 1, crop_px, crop_px), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = _session(path).run(None, {"eye": x})[0]
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    return max_abs_diff(ref, got), agree


def eval_on_test_split(paths: dict, crop_px: int) -> Optional[dict]:
    """Accuracy of each exported graph on the held-out MRL subjects.

    `verify()` compares against RANDOM inputs, which proves the graph was
    translated faithfully but says nothing about whether quantization cost
    accuracy on real eyes. Only this does. Returns None when the dataset is not
    present, so the export still works without it.
    """
    try:
        from . import mrl
        from .config import MRL_DIR, TrainConfig
        from .data import EyeCropDataset

        cfg = TrainConfig()
        samples = mrl.scan(MRL_DIR)
        _, _, test = mrl.split_by_subject(samples, cfg.val_frac, cfg.test_frac, cfg.seed)
    except (FileNotFoundError, ImportError):
        return None

    ds = EyeCropDataset(test, crop_px)
    x = (ds._cache.astype(np.float32) / 255.0)[:, None]
    y = np.array([s.label for s in test])

    out = {"n": int(len(y)), "subjects": len(({s.subject for s in test}))}
    for name, path in paths.items():
        sess = _session(path)
        preds = []
        for i in range(0, len(x), 512):
            preds.append(sess.run(None, {"eye": x[i:i + 512]})[0])
        out[name] = float((np.concatenate(preds).argmax(1) == y).mean())
    return out


def bench(path: Path, crop_px: int, iters: int = 200, batch: int = 2) -> dict:
    """Single-threaded CPU latency. A Pi 5 is slower than this machine —
    treat the number as a floor to measure against, not a prediction."""
    sess = _session(path)
    x = np.random.rand(batch, 1, crop_px, crop_px).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"eye": x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"eye": x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "batch": batch,
        "p50_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export eye-state model to ONNX + int8.")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--calib-dir", type=str, default=None,
                    help="directory of real eye crops for quantization calibration")
    ap.add_argument("--bench", type=int, default=200, help="latency iterations (0 to skip)")
    args = ap.parse_args()

    _require_onnx()
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint) if args.checkpoint else out_dir / "eyestate.pt"

    model, crop_px = load_model(ckpt)
    print(f"loaded {ckpt}  ({model.n_params():,} params, {crop_px}x{crop_px} input)")

    fp32 = out_dir / "eyestate_fp32.onnx"
    export_fp32(model, crop_px, fp32)
    d, agree = verify(model, fp32, crop_px)
    print(f"fp32 onnx   {fp32.stat().st_size / 1024:7.1f} KB   "
          f"max|Δlogit| {d:.2e}   argmax agreement {agree:.3f}")
    if d > 1e-3:
        raise SystemExit(f"fp32 export diverges from torch (max diff {d:.3e}) — do not ship this.")

    int8 = out_dir / "eyestate_int8.onnx"
    calib = Path(args.calib_dir) if args.calib_dir else None
    if calib is None:
        print("  no --calib-dir: calibrating on synthetic eyes. "
              "Re-run with real crops before trusting the int8 accuracy.")
    quantize_int8(fp32, int8, crop_px, calib)
    d8, agree8 = verify(model, int8, crop_px)
    print(f"int8 onnx   {int8.stat().st_size / 1024:7.1f} KB   "
          f"max|Δlogit| {d8:.2e}   argmax agreement {agree8:.3f}")

    report = {
        "checkpoint": str(ckpt),
        "crop_px": crop_px,
        "n_params": model.n_params(),
        "fp32": {"path": str(fp32), "kb": round(fp32.stat().st_size / 1024, 1),
                 "max_logit_diff": d, "argmax_agreement": agree},
        "int8": {"path": str(int8), "kb": round(int8.stat().st_size / 1024, 1),
                 "max_logit_diff": d8, "argmax_agreement": agree8,
                 "calibrated_on": str(calib) if calib else "synthetic"},
    }

    acc = eval_on_test_split({"fp32": fp32, "int8": int8}, crop_px)
    if acc:
        report["test_accuracy"] = acc
        print(f"accuracy on {acc['n']} images from {acc['subjects']} unseen subjects")
        print(f"  fp32  {acc['fp32']:.4f}")
        print(f"  int8  {acc['int8']:.4f}   "
              f"({acc['int8'] - acc['fp32']:+.4f} vs fp32)")
    else:
        print("  MRL not present — skipping real-data accuracy check")

    if args.bench:
        report["bench_fp32"] = bench(fp32, crop_px, args.bench)
        report["bench_int8"] = bench(int8, crop_px, args.bench)
        print(f"latency (1 thread, batch 2 = both eyes)")
        print(f"  fp32  p50 {report['bench_fp32']['p50_ms']:.3f} ms  "
              f"p95 {report['bench_fp32']['p95_ms']:.3f} ms")
        print(f"  int8  p50 {report['bench_int8']['p50_ms']:.3f} ms  "
              f"p95 {report['bench_int8']['p95_ms']:.3f} ms")

    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out_dir / 'export_report.json'}")

    if agree8 < 0.98:
        print("\nWARNING: int8 disagrees with fp32 on more than 2% of inputs. "
              "Calibrate on real crops, or ship fp32 if the budget allows it.")


if __name__ == "__main__":
    main()
