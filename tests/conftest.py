import pytest
import torch


@pytest.fixture(params=[1, 2, 8, 16, 50])
def batch_size(request):
    return request.param


@pytest.fixture(params=[16, 32, 128])  # must be an exact multiple of num_heads
def D(request):
    return request.param


@pytest.fixture(params=[16*4, 32*4, 100*4])
def D_ff(request):
    return request.param


@pytest.fixture(params=[3, 6])
def num_layers(request):
    return request.param


@pytest.fixture(params=[4, 8])
def num_heads(request):
    return request.param


@pytest.fixture(params=[7, 13, 22])
def input_sequence_length(request):
    return request.param  # random sequences


@pytest.fixture(params=[6, 14, 23])
def target_sequence_length(request):
    return request.param  # random sequences


@pytest.fixture
def src_vocab_size():
    return 123  # random


@pytest.fixture
def trg_vocab_size():
    return 98  # random


@pytest.fixture
def src_mask_tensor(batch_size, input_sequence_length):
    return torch.rand(batch_size, 1, input_sequence_length) > 0.5


@pytest.fixture
def trg_mask_tensor(batch_size, target_sequence_length):
    return torch.rand(batch_size, target_sequence_length, target_sequence_length) > 0.5


@pytest.fixture
def input_sequence(batch_size, input_sequence_length):
    return (10 * torch.rand(batch_size, input_sequence_length)).long()


@pytest.fixture
def target_sequence(batch_size, target_sequence_length):
    return (10 * torch.rand(batch_size, target_sequence_length)).long()
