import torch
import torch.nn as nn

from src.losses.anticollapse import EppsPulley, BCS
from src.losses.prediction import TGJEPALoss


def test_smoothl1_zero_when_equal():
    loss_fn = TGJEPALoss(lambda_reg=0.0)
    z = torch.randn(10, 256)
    # dummy full-batch tensors for sigreg (same z used for both online and target all)
    total, pred_loss, sigreg_dict = loss_fn(z, z, z, z)
    assert pred_loss.item() < 1e-6, \
        f"prediction loss should be 0 when pred==target, got {pred_loss.item()}"


def test_bcs_returns_expected_keys():
    bcs = BCS()
    z1 = torch.randn(32, 256)
    z2 = torch.randn(32, 256)
    result = bcs(z1, z2)
    assert isinstance(result, dict), "BCS.forward must return a dict"
    assert set(result.keys()) == {'loss', 'bcs_loss', 'invariance_loss'}, \
        f"unexpected keys: {set(result.keys())}"


def test_bcs_detects_collapse():
    bcs = BCS()
    z_random = torch.randn(32, 256)
    z_collapsed = torch.ones(32, 256)  # all same embedding
    loss_random = bcs(z_random, z_random)['loss'].item()
    loss_collapsed = bcs(z_collapsed, z_collapsed)['loss'].item()
    assert loss_collapsed > loss_random, \
        "BCS loss should be higher for collapsed embeddings than random embeddings"


def test_total_loss_structure():
    lmbd = 0.01
    loss_fn = TGJEPALoss(lambda_reg=lmbd)
    z_pred = torch.randn(10, 256)
    z_target = torch.randn(10, 256)
    z_all = torch.randn(32, 256)

    total, pred_loss, sigreg_dict = loss_fn(z_pred, z_target, z_all, z_all)

    expected_total = pred_loss + lmbd * sigreg_dict['loss']
    assert torch.allclose(total, expected_total, atol=1e-5), \
        "total loss must equal pred_loss + lambda_reg * sigreg_dict['loss']"


def test_lambda_reg_effect():
    z_pred = torch.randn(10, 256)
    z_target = torch.randn(10, 256)
    z_all = torch.randn(32, 256)

    loss_low = TGJEPALoss(lambda_reg=0.001)(z_pred, z_target, z_all, z_all)[0]
    loss_high = TGJEPALoss(lambda_reg=1.0)(z_pred, z_target, z_all, z_all)[0]

    assert loss_high > loss_low, \
        "higher lambda_reg should yield higher total loss when sigreg loss is nonzero"


def test_epps_pulley_is_module():
    ep = EppsPulley()
    assert isinstance(ep, nn.Module), "EppsPulley must be an nn.Module"
    # verify it is callable (has __call__)
    z = torch.randn(16, 256)
    out = ep(z)
    assert isinstance(out, torch.Tensor), "EppsPulley(z) must return a Tensor"


def test_no_bare_epps_pulley_function():
    import src.losses.anticollapse as module
    # there must be no bare function called epps_pulley (only the class EppsPulley)
    bare_fn = getattr(module, 'epps_pulley', None)
    if bare_fn is not None:
        assert not callable(bare_fn) or isinstance(bare_fn, type), \
            "epps_pulley should not exist as a bare function; use EppsPulley class instead"
