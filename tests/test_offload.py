import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from turkmenocr.offload import CpuExpert, module_bytes, place_model  # noqa: E402


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class MoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(4)])
        self.gate = nn.Linear(4, 4, bias=False)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MoE() for _ in range(2)])


def test_nothing_offloaded_when_budget_suffices():
    model = Model().half()
    assert place_model(model, "cpu", module_bytes(model)) == (0, 8)


def test_experts_spread_across_layers():
    model = Model().half()
    one_expert = module_bytes(model.layers[0].experts[0])
    offloaded, total = place_model(model, "cpu", module_bytes(model) - 2 * one_expert)
    assert (offloaded, total) == (2, 8)
    assert isinstance(model.layers[0].experts[3], CpuExpert)
    assert isinstance(model.layers[1].experts[3], CpuExpert)
    assert not isinstance(model.layers[0].experts[2], CpuExpert)


def test_cpu_expert_matches_original_and_keeps_dtype():
    expert = Expert()
    reference = expert(torch.ones(2, 4))
    wrapped = CpuExpert(Expert())
    wrapped.expert.load_state_dict(expert.state_dict())
    output = wrapped(torch.ones(2, 4, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert torch.allclose(output.float(), reference, atol=1e-2)
    assert list(wrapped.parameters()) == []
