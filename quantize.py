"""Post-training static INT8 quantization of the ONNX model with ONNX Runtime.

Calibration uses a small slice of the *training* set, so the test set stays untouched.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import ONNX_FP32, ONNX_INT8, load_cifar10


class CifarCalibrationReader(CalibrationDataReader):
    def __init__(self, n_samples: int, batch_size: int = 64, seed: int = 0):
        x, _ = load_cifar10(train=True)
        idx = np.random.default_rng(seed).choice(len(x), n_samples, replace=False)
        x = x[idx]
        self._batches = iter(
            {"input": x[i : i + batch_size]} for i in range(0, n_samples, batch_size)
        )

    def get_next(self):
        return next(self._batches, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--per-tensor", action="store_true",
                    help="use per-tensor instead of per-channel weight quantization")
    ap.add_argument("--reduce-range", action="store_true",
                    help="quantize weights to 7 bits (avoids u8*s8 saturation on x86 CPUs "
                         "without VNNI)")
    ap.add_argument("--method",choices=["minmax", "entropy", "percentile"], default="minmax",
                    help="how activation ranges are picked from the calibration data")
    ap.add_argument("--out", type=Path, default=ONNX_INT8)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prep.onnx"
        # Shape inference + graph optimization so every tensor can get a quantizer.
        quant_pre_process(str(ONNX_FP32), str(prepped))
        quantize_static(
            model_input=str(prepped),
            model_output=str(args.out),
            calibration_data_reader=CifarCalibrationReader(args.calib_samples),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=not args.per_tensor,
            reduce_range=args.reduce_range,
            calibrate_method={
                "minmax": CalibrationMethod.MinMax,
                "entropy": CalibrationMethod.Entropy,
                "percentile": CalibrationMethod.Percentile,
            }[args.method],
        )

    fp32_kb = ONNX_FP32.stat().st_size / 1024
    int8_kb = args.out.stat().st_size / 1024
    print(f"fp32: {fp32_kb:.0f} KB -> int8: {int8_kb:.0f} KB  ({fp32_kb / int8_kb:.2f}x smaller)")


if __name__ == "__main__":
    main()
