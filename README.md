# CNN → ONNX → INT8: measuring what quantization buys on a real CPU

A small end-to-end optimization demo: train a CNN in PyTorch, export it to ONNX, run it
with ONNX Runtime, apply post-training INT8 quantization, and measure accuracy, size and
latency before and after.

The interesting part is not the speed-up. The first INT8 model lost 3.5 points of accuracy,
and the cause turned out to be the CPU's instruction set, not the quantization itself.

## Results

ResNet-8-style CNN (0.3M parameters) on CIFAR-10, 10 epochs of training.
Measured on an AMD Ryzen 5 5600H (6 cores, AVX2, no VNNI), Windows 11, ONNX Runtime 1.31.1,
CPU execution provider.

| Metric | FP32 | INT8, 8-bit weights | INT8, 7-bit weights |
|---|---|---|---|
| Top-1 accuracy (10k test images) | 86.48% | 82.93% (-3.55 pp) | **86.50% (+0.02 pp)** |
| Same model in exact arithmetic | - | 86.50% | 86.50% |
| Predictions matching FP32 | - | 90.78% | 98.87% |
| Model size | 1207 KB | 330 KB (3.65x smaller) | 330 KB (3.65x smaller) |
| Latency, batch 1, 1 thread, median | 0.897 ms | 0.558 ms (1.61x) | 0.560 ms (1.60x) |
| Latency, batch 1, 1 thread, p95 | 0.967 ms | 0.615 ms (1.57x) | 0.619 ms (1.56x) |
| Latency, batch 1, all threads, median | 0.376 ms | 0.257 ms (1.46x) | 0.257 ms (1.46x) |
| Latency, batch 1, all threads, p95 | 0.462 ms | 0.324 ms (1.43x) | 0.307 ms (1.50x) |

Raw numbers are in [`results.json`](results.json).

**Bottom line:** with 7-bit weights the INT8 model is 3.65x smaller and about 1.5x faster
at the same accuracy as FP32.

## The hardware-aware part

The default recipe (per-channel, 8-bit signed weights, 8-bit unsigned activations) cost
3.55 points of accuracy. Three checks narrowed down why:

1. **Calibration was not the cause.** MinMax, entropy and percentile calibration, and 4x
   more calibration images, all gave 82.9%.
2. **The quantization parameters were fine.** Running the same INT8 model with ONNX Runtime
   graph optimizations disabled keeps the Q/DQ nodes unfused, so the convolutions run in
   float on the dequantized weights. That gives 86.50%. The loss only appears when the
   integer kernels are used.
3. **Reducing weights to 7 bits fixed it.** `reduce_range=True` brings the integer-kernel
   accuracy to 86.50%, identical to the exact-arithmetic figure.

This matches a known x86 limitation. Without VNNI, the u8 x s8 multiply-add
(`vpmaddubsw`) sums two products into a saturating 16-bit integer, so large activations
times large weights can clip. Per-channel quantization makes this more likely because
every channel's weights are stretched to the full ±127 range (per-tensor weights lost only
0.43 points here). On CPUs with VNNI, or on ARM, the 8-bit model should not need this
workaround; I have not measured that.

The takeaway: the same ONNX file can give different accuracy on different hardware, so
quantized accuracy has to be measured on the target, with the kernels that will actually
run.

## How it is measured

- **Accuracy:** full 10,000-image CIFAR-10 test set. Calibration uses 512 training images,
  so the test set is never seen during quantization.
- **Latency:** batch size 1, 5,000 timed runs per model after 100 warm-up runs. The models
  are timed in alternating rounds so background load and thermal drift affect them equally.
- **Export check:** ONNX Runtime output matches PyTorch to 1.7e-6 on a random batch.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python train.py                                      # ~20 min on a laptop CPU
python export_onnx.py                                # models/cnn_fp32.onnx
python quantize.py                                   # models/cnn_int8.onnx
python quantize.py --reduce-range --out models/cnn_int8_rr.onnx
python benchmark.py                                  # prints the table, writes results.json
```

The trained checkpoint and all three ONNX models are committed in `models/`, so
`benchmark.py` can be run directly without training.

| File | Purpose |
|---|---|
| `common.py` | Model definition and CIFAR-10 loading |
| `train.py` | Training loop (SGD, one-cycle LR, crop + flip augmentation) |
| `export_onnx.py` | ONNX export with a PyTorch-vs-ONNX Runtime parity check |
| `quantize.py` | Static post-training quantization (QDQ format); options for calibration method, per-tensor weights and reduced range |
| `benchmark.py` | Accuracy, size and latency comparison |

## Limitations

- One CPU, one model, one dataset. Latency ratios will differ on other hardware.
- At under 1 ms per inference, a meaningful share of the time is ONNX Runtime call
  overhead, which quantization does not reduce. Larger models typically show a bigger
  speed-up.
- Post-training quantization only; quantization-aware training was not needed here.
