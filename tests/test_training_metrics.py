from types import SimpleNamespace

import lightning as L
import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, roc_auc_score

from mole_classificator.mole_classificator import Classificator


@pytest.fixture
def make_model(monkeypatch):
    # These checks exercise metric/checkpoint behavior without a pretrained backbone.
    monkeypatch.setattr("torchvision.models.convnext_tiny", lambda **kwargs: SimpleNamespace(
        features=torch.nn.Sequential(torch.nn.Conv2d(3, 768, 1)),
    ))

    def create(**params):
        return Classificator(lr=1e-3, weight_decay=1e-5, num_classes=3,
                             pretrained=False, d_model=8, nhead=2, num_layers=1,
                             num_tokens=1, **params)

    return create


@pytest.mark.parametrize("stage,hook", [("val", "validation"), ("test", "test")])
def test_epoch_metrics_match_sklearn_with_unequal_batches_and_reset(make_model, monkeypatch, stage, hook):
    model = make_model(class_names=["a", "b", "c"])
    logged = {}
    monkeypatch.setattr(model, "forward", lambda logits: logits)
    monkeypatch.setattr(model, "log", lambda name, value, **kwargs: logged.update({name: value.detach().clone()}))
    probabilities = torch.tensor([
        [.7, .2, .1], [.2, .7, .1], [.6, .3, .1], [.2, .5, .3],
        [.1, .3, .6], [.2, .5, .3], [.2, .1, .7],
    ])
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    predictions = probabilities.argmax(-1)
    expected_f1 = f1_score(labels, predictions, labels=[0, 1, 2], average=None, zero_division=0)
    expected_auc = roc_auc_score(labels, probabilities, labels=[0, 1, 2], average=None, multi_class="ovr")
    batch_f1 = [f1_score(labels[part], predictions[part], labels=[0, 1, 2],
                         average="macro", zero_division=0) for part in (slice(0, 2), slice(2, None))]
    assert np.mean(batch_f1) != pytest.approx(expected_f1.mean())

    start = getattr(model, f"on_{hook}_epoch_start")
    step = getattr(model, f"{hook}_step")
    end = getattr(model, f"on_{hook}_epoch_end")
    start()
    for batch_idx, part in enumerate((slice(0, 2), slice(2, None))):
        step((probabilities[part].log(), labels[part]), batch_idx)
    end()
    assert logged[f"{stage}_f1_macro"].item() == pytest.approx(expected_f1.mean())
    assert logged[f"{stage}_auroc_macro"].item() == pytest.approx(expected_auc.mean())
    for index, name in enumerate(["a", "b", "c"]):
        assert logged[f"{stage}_f1/{name}"].item() == pytest.approx(expected_f1[index])
        assert logged[f"{stage}_auroc/{name}"].item() == pytest.approx(expected_auc[index])

    # A second, perfect epoch must not retain any first-epoch predictions.
    start()
    step((torch.eye(3).mul(5), torch.arange(3)), 0)
    end()
    assert logged[f"{stage}_f1_macro"].item() == pytest.approx(1)
    assert logged[f"{stage}_auroc_macro"].item() == pytest.approx(1)


def test_validation_and_test_have_separate_states_and_numeric_class_names(make_model, monkeypatch):
    model = make_model()
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kwargs: logged.update({name: value.detach().clone()}))
    labels = torch.arange(3)
    perfect = torch.eye(3)
    wrong = torch.roll(perfect, 1, dims=1)
    model.val_f1.update(perfect, labels)
    model.val_auroc.update(perfect, labels)
    model.test_f1.update(wrong, labels)
    model.test_auroc.update(wrong, labels)
    model.on_validation_epoch_end()
    model.on_test_epoch_end()
    assert logged["val_f1_macro"].item() == pytest.approx(1)
    assert logged["test_f1_macro"].item() == pytest.approx(0)
    assert set(f"val_f1/{index}" for index in range(3)).issubset(logged)


@pytest.mark.parametrize("weights", [None, [1.0, 2.0, 3.0]])
def test_metrics_are_nonpersistent_and_weighted_checkpoints_load_strictly(make_model, tmp_path, weights):
    params = {"class_weights": weights} if weights is not None else {}
    model = make_model(**params)
    model.val_f1.update(torch.eye(3), torch.arange(3))
    model.val_auroc.update(torch.eye(3), torch.arange(3))
    state = model.state_dict()
    assert not any(name.startswith(("val_f1.", "val_auroc.", "test_f1.", "test_auroc.")) for name in state)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": state, "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": L.__version__}, checkpoint)
    restored = Classificator.load_from_checkpoint(checkpoint, pretrained=False, strict=True)
    if weights is None:
        assert restored.loss_fn.weight is None
        assert "loss_fn.weight" not in state
    else:
        torch.testing.assert_close(restored.loss_fn.weight, torch.tensor(weights))
        logits = torch.tensor([[2., 0., 1.], [0., 1., 2.]])
        labels = torch.tensor([0, 1])
        torch.testing.assert_close(restored.loss_fn(logits, labels), model.loss_fn(logits, labels))
    torch.testing.assert_close(restored.pos_embedding, model.pos_embedding)


def test_backbone_stays_registered_for_ddp_but_updates_only_after_freezing(make_model):
    model = make_model(diagnostics_every_n_steps=0)
    model._trainer = SimpleNamespace(current_epoch=0, max_epochs=4)
    try:
        optimizer = model.configure_optimizers()["optimizer"]
        parameter = next(model.backbone.parameters())
        assert all(item.requires_grad for item in model.backbone.parameters())
        images = torch.rand(2, 3, 4, 4)
        for epoch, should_update in ((0, False), (3, True)):
            model._trainer.current_epoch = epoch
            model.on_train_epoch_start()
            assert parameter.requires_grad
            assert model.backbone.training == should_update
            before = parameter.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            model.backbone(images).square().mean().backward()
            assert parameter.grad is not None
            model.on_before_optimizer_step(optimizer)
            assert (parameter.grad is not None) == should_update
            optimizer.step()
            assert torch.equal(parameter, before) != should_update
    finally:
        model._trainer = None
