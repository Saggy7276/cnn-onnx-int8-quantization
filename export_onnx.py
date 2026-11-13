"""Export the trained PyTorch model to ONNX and check it matches PyTorch under ONNX Runtime."""
import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import CKPT, ONNX_FP32, SmallResNet


def main():
    model = SmallResNet()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_FP32),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,  # folds BatchNorm into the preceding Conv
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(ONNX_FP32)))

    # Parity check: same random batch through PyTorch and ONNX Runtime.
    x = torch.randn(16, 3, 32, 32)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(ONNX_FP32), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())
    print(f"exported {ONNX_FP32.name}  ({ONNX_FP32.stat().st_size / 1024:.0f} KB)")
    print(f"max |pytorch - onnxruntime| = {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX output diverges from PyTorch"


if __name__ == "__main__":
    main()
