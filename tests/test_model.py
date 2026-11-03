import torch
from src.data import MEAN, STD, denormalize, get_transforms
from src.model import CLASSES, CNN, count_parameters


def test_output_shape_and_raw_logits():
    model = CNN().eval()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)
    out = model(torch.randn(64, 3, 32, 32))
    assert (out < 0).any(), "logits should not be clipped by a final ReLU"


def test_backward_pass_updates_weights():
    model = CNN()
    before = model.classifier[-1].weight.clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 3, 32, 32)),
                                             torch.randint(0, 10, (8,)))
    loss.backward()
    opt.step()
    assert not torch.equal(before, model.classifier[-1].weight)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    model = CNN(p_drop=0.0)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.5


def test_parameter_count_reasonable():
    assert 100_000 < count_parameters(CNN()) < 1_000_000


def test_transform_shapes_and_denormalize_roundtrip():
    from PIL import Image
    img = Image.new("RGB", (32, 32), (120, 60, 200))
    t = get_transforms(augment=True)(img)
    assert t.shape == (3, 32, 32)
    flat = get_transforms(augment=False)(img)
    back = denormalize(flat)
    assert back.shape == (32, 32, 3)
    assert torch.allclose(back[0, 0], torch.tensor([120, 60, 200]) / 255.0, atol=1e-4)


def test_constants():
    assert len(CLASSES) == 10 and len(MEAN) == 3 and len(STD) == 3
