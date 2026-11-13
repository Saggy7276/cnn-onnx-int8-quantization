"""Measure accuracy, latency and size of the FP32 and INT8 ONNX models under ONNX Runtime."""
import argparse
import json
import platform
import time

import numpy as np
import onnxruntime as ort

from common import ONNX_FP32, ONNX_INT8, ONNX_INT8_RR, ROOT, load_cifar10

OPT_ALL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
OPT_NONE = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

MODELS = {
    "fp32": ("FP32", ONNX_FP32),
    "int8": ("INT8 (8-bit weights)", ONNX_INT8),
    "int8_rr": ("INT8 (7-bit weights)", ONNX_INT8_RR),
}


def make_session(path, threads: int, opt_level=OPT_ALL):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = opt_level
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def predict(path, x, batch_size=500, opt_level=OPT_ALL):
    sess = make_session(path, threads=0, opt_level=opt_level)  # 0 = let ORT use all cores
    preds = [
        sess.run(None, {"input": x[i : i + batch_size]})[0].argmax(1)
        for i in range(0, len(x), batch_size)
    ]
    return np.concatenate(preds)


def time_runs(sess, x, n):
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        out[i] = time.perf_counter() - t0
    return out * 1e3  # ms


def latency(paths, x, threads, runs, rounds=5, warmup=100):
    """Batch-1 latency. Models are timed in alternating rounds so that thermal or
    background-load drift hits all of them equally."""
    sessions = {name: make_session(p, threads) for name, p in paths.items()}
    for s in sessions.values():
        time_runs(s, x, warmup)
    samples = {name: [] for name in paths}
    for _ in range(rounds):
        for name, s in sessions.items():
            samples[name].append(time_runs(s, x, runs // rounds))
    stats = {}
    for name, chunks in samples.items():
        t = np.concatenate(chunks)
        stats[name] = {
            "mean_ms": float(t.mean()),
            "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)),
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5000, help="timed inferences per model")
    args = ap.parse_args()

    paths = {name: p for name, (_, p) in MODELS.items()}
    x_te, y_te = load_cifar10(train=False)

    results = {
        "env": {
            "cpu": platform.processor(),
            "os": f"{platform.system()} {platform.release()}",
            "onnxruntime": ort.__version__,
            "test_images": int(len(x_te)),
            "timed_runs_per_model": args.runs,
        },
        "models": {},
    }

    preds = {name: predict(p, x_te) for name, p in paths.items()}
    for name, p in paths.items():
        m = {
            "size_kb": p.stat().st_size / 1024,
            "top1_acc": float((preds[name] == y_te).mean()),
            "agreement_with_fp32": float((preds[name] == preds["fp32"]).mean()),
        }
        if name != "fp32":
            # With graph optimizations off, ORT does not fuse the Q/DQ nodes into integer
            # kernels: the convolutions run in float on the dequantized weights. This is the
            # accuracy the quantization parameters give in exact arithmetic, so any gap to
            # top1_acc comes from the integer kernels, not from quantization error.
            ref = predict(p, x_te, opt_level=OPT_NONE)
            m["top1_acc_exact_arithmetic"] = float((ref == y_te).mean())
        results["models"][name] = m

    sample = x_te[:1]
    for label, threads in (("1_thread", 1), ("all_threads", 0)):
        for name, s in latency(paths, sample, threads, args.runs).items():
            results["models"][name][f"latency_{label}"] = s

    (ROOT / "results.json").write_text(json.dumps(results, indent=2))

    r = results["models"]
    f = r["fp32"]
    quant = [n for n in MODELS if n != "fp32"]

    def row(title, base, cells):
        return "| " + " | ".join([title, base, *cells]) + " |"

    lines = [
        "| Metric | " + " | ".join(t for t, _ in MODELS.values()) + " |",
        "|---|" + "---|" * len(MODELS),
        row("Top-1 accuracy (10k test images)", f"{f['top1_acc']:.2%}",
            [f"{r[n]['top1_acc']:.2%} ({(r[n]['top1_acc'] - f['top1_acc']) * 100:+.2f} pp)"
             for n in quant]),
        row("Same accuracy in exact arithmetic", "-",
            [f"{r[n]['top1_acc_exact_arithmetic']:.2%}" for n in quant]),
        row("Predictions matching FP32", "-",
            [f"{r[n]['agreement_with_fp32']:.2%}" for n in quant]),
        row("Model size", f"{f['size_kb']:.0f} KB",
            [f"{r[n]['size_kb']:.0f} KB ({f['size_kb'] / r[n]['size_kb']:.2f}x smaller)"
             for n in quant]),
    ]
    for label, title in (("1_thread", "1 thread"), ("all_threads", "all threads")):
        for key, stat in (("p50_ms", "median"), ("p95_ms", "p95")):
            a = f[f"latency_{label}"][key]
            lines.append(row(
                f"Latency, batch 1, {title}, {stat}", f"{a:.3f} ms",
                [f"{r[n][f'latency_{label}'][key]:.3f} ms "
                 f"({a / r[n][f'latency_{label}'][key]:.2f}x)" for n in quant]))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
