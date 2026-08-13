"""Checks on the exported ONNX artifact.

Skips cleanly when model.onnx has not been exported, so the suite still runs on
a machine that has only done `fetch_model.py`.

    uv run python scripts/export_onnx.py

These run ONNX Runtime on the CPU provider deliberately. The question here is
whether the *export* preserved the model, not how fast any runtime is; pinning
the provider keeps cuDNN's algorithm choice out of the comparison.
"""

import numpy as np
import pytest
import torch

from src.benchmark import compare_logits
from src.config import Settings
from src.model.pytorch_model import WEIGHTS_FILE, load_from_repository

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

# Taken off the module rather than imported: a top-level import would run
# before importorskip and break the suite on a machine without onnxruntime.
InvalidArgument = ort.capi.onnxruntime_pybind11_state.InvalidArgument

INPUT_NAME = "input"


@pytest.fixture(scope="module")
def paths():
    s = Settings(_env_file=None)
    model_path = s.model_dir / "model.onnx"
    if not model_path.is_file():
        pytest.skip("no model.onnx: uv run python scripts/export_onnx.py")
    if not (s.model_dir / WEIGHTS_FILE).is_file():
        pytest.skip("model not provisioned")
    return s, model_path


@pytest.fixture(scope="module")
def graph(paths):
    _, model_path = paths
    return onnx.load(str(model_path))


@pytest.fixture(scope="module")
def session(paths):
    _, model_path = paths
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def test_graph_is_structurally_valid(paths):
    _, model_path = paths
    onnx.checker.check_model(str(model_path), full_check=True)


def test_artifact_is_self_contained(paths):
    """No sibling .data file.

    A model.onnx that silently depends on model.onnx.data is a deployment
    footgun: it copies, it loads, and it fails at the first inference.
    """
    _, model_path = paths
    assert not model_path.with_suffix(".onnx.data").exists()
    assert model_path.stat().st_size > 40 * 1024 * 1024  # weights really are inside


def test_batch_dimension_is_symbolic(graph):
    """The whole point of the export. A fixed 1 here makes Phase 11 impossible."""
    inp = next(v for v in graph.graph.input if v.name == INPUT_NAME)
    dims = inp.type.tensor_type.shape.dim
    assert dims[0].dim_param != "", "batch dimension must be symbolic, not fixed"
    assert [d.dim_value for d in dims[1:]] == [3, 224, 224]


def test_batchnorm_was_folded_into_conv(graph):
    """Export already fused BatchNorm away; ResNet-18 has 20 of them.

    Recorded so a later 'TensorRT fused our BatchNorms' claim cannot be made:
    they were gone before any runtime saw the graph.
    """
    ops = [n.op_type for n in graph.graph.node]
    assert ops.count("Conv") == 20
    assert ops.count("BatchNormalization") == 0


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_dynamic_batch_actually_works(session, n):
    # Declaring a dynamic axis and accepting one are different claims.
    out = session.run(None, {INPUT_NAME: np.zeros((n, 3, 224, 224), dtype=np.float32)})[0]
    assert out.shape == (n, 1000)


def test_onnx_agrees_with_pytorch(paths, session):
    s, _ = paths
    data = np.random.default_rng(0).standard_normal((4, 3, 224, 224), dtype=np.float32)

    model = load_from_repository(s.model_dir, s.model_name).eval().cpu()
    with torch.inference_mode():
        expected = model(torch.from_numpy(data)).numpy()

    got = session.run(None, {INPUT_NAME: data})[0]
    agreement = compare_logits(expected, got)

    # An export that traced training-mode BatchNorm, or dropped an op the tracer
    # could not see, passes onnx.checker and fails right here.
    assert agreement.top1_agreement == 1.0
    assert agreement.max_abs_logit_diff < 1e-4


def test_wrong_input_shape_is_rejected(session):
    # Height and width are fixed in the graph, so 256x256 must fail loudly here
    # rather than reach a kernel. Asserting the specific type, not bare
    # Exception: a test that passes on any failure also passes when the session
    # is broken for an entirely unrelated reason.
    with pytest.raises(InvalidArgument, match="224"):
        session.run(None, {INPUT_NAME: np.zeros((1, 3, 256, 256), dtype=np.float32)})


def test_wrong_dtype_is_rejected(session):
    # float64 is numpy's default, so this is the single easiest mistake to make.
    with pytest.raises(InvalidArgument, match="(?i)type"):
        session.run(None, {INPUT_NAME: np.zeros((1, 3, 224, 224), dtype=np.float64)})
