import pytest

from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.cuda_specs import CUDASpecs


@pytest.fixture
def hip_spec() -> CUDASpecs:
    # All dummy values to pass test. This mainly test if libbitsandbytes_hip_nohipblaslt.so exist in bitsandbytes dir
    return CUDASpecs(
        cuda_version_string="120",
        highest_compute_capability=(8, 6),
        cuda_version_tuple=(12, 0),
    )

def test_get_cuda_bnb_library_path(monkeypatch, hip_spec):
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    assert get_cuda_bnb_library_path(hip_spec).stem == "libbitsandbytes_hip_nohipblaslt"


#def test_get_cuda_bnb_library_path_override(monkeypatch, cuda120_spec, caplog):
#    monkeypatch.setenv("BNB_CUDA_VERSION", "110")
#    assert get_cuda_bnb_library_path(cuda120_spec).stem == "libbitsandbytes_cuda110"
#    assert "BNB_CUDA_VERSION" in caplog.text  # did we get the warning?


#def test_get_cuda_bnb_library_path_nocublaslt(monkeypatch, cuda111_noblas_spec):
#    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
#    assert get_cuda_bnb_library_path(cuda111_noblas_spec).stem == "libbitsandbytes_cuda111_nocublaslt"
