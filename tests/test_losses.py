import pytest
import torch

from model_training.common.losses import loss


pytestmark = pytest.mark.losses

standard_approx = 0.000001


@pytest.fixture(scope="module")
def tensor1():
    """Shapes of tensors should be: [batch_size, channels, width, height]"""
    yield torch.tensor([[[[1, 1], [0, 0]]]]).float()


@pytest.fixture(scope="module")
def tensor2():
    """Shapes of tensors should be: [batch_size, channels, width, height]"""
    yield torch.tensor([[[[0, 0], [1, 1]]]]).float()


@pytest.fixture(scope="module")
def tensor3():
    """Shapes of tensors should be: [batch_size, channels, width, height]"""
    yield torch.tensor(
        [[[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]]]
    ).float()


@pytest.fixture(scope="module")
def tensor4():
    """Shapes of tensors should be: [batch_size, channels, width, height]"""
    yield torch.tensor(
        [[[[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]]]
    ).float()


def test_getting_correct_loss():
    assert loss.get_loss({"name": "lovasz"})


def test_getting_incorrect_loss():
    with pytest.raises(ValueError):
        assert loss.get_loss({"name": "this_is_some_incorrect_name_for_loss_function"})


def test_iou_loss(tensor1, tensor2, tensor3, tensor4):
    iou_loss = loss.get_loss({"name": "iou_loss"})
    assert iou_loss.iou_metric(tensor1, tensor1) == 1.0
    assert iou_loss.iou_metric(tensor2, tensor2) == 1.0

    # almost zero
    iou_result = iou_loss.iou_metric(tensor1, tensor2)
    assert abs(iou_result) < standard_approx

    assert iou_loss.iou_metric(tensor3, tensor4).data.numpy() == pytest.approx(
        0.3333333, standard_approx
    )


def test_mixed_loss(tensor1, tensor2, tensor3, tensor4):
    mixed_loss = loss.get_loss({"name": "mixed_loss"})

    assert mixed_loss.forward(tensor1, tensor1).data.numpy() == pytest.approx(
        1.3103894, standard_approx
    )
    assert mixed_loss.forward(tensor2, tensor2).data.numpy() == pytest.approx(
        1.3103894, standard_approx
    )
    assert mixed_loss.forward(tensor1, tensor2).data.numpy() == pytest.approx(
        4.975, standard_approx
    )
    assert mixed_loss.forward(tensor3, tensor4).data.numpy() == pytest.approx(
        3.230158, standard_approx
    )


def test_dice_loss(tensor1, tensor2, tensor3, tensor4):
    dice_loss = loss.get_loss({"name": "dice_loss"})

    assert dice_loss.forward(tensor1, tensor1).data.numpy() == pytest.approx(
        0.71844566, standard_approx
    )
    assert dice_loss.forward(tensor2, tensor2).data.numpy() == pytest.approx(
        0.71844566, standard_approx
    )
    assert dice_loss.forward(tensor1, tensor2).data.numpy() == pytest.approx(
        0.54923755, standard_approx
    )
    assert dice_loss.forward(tensor3, tensor4).data.numpy() == pytest.approx(
        0.57556236, standard_approx
    )


# def test_espnet_crossentropy_loss(tensor3, tensor4):
#     espnet_crossentropy_loss = loss.get_loss({"name": "espnet_crossentropy_loss"})
#
#     # assert espnet_crossentropy_loss.forward(tensor1, tensor1).data.numpy() == pytest.approx(0.71844566, standard_approx)
#     # assert espnet_crossentropy_loss.forward(tensor2, tensor2).data.numpy() == pytest.approx(0.71844566, standard_approx)
#     assert espnet_crossentropy_loss.forward(tensor1, tensor2).data.numpy() == pytest.approx(0.54923755, standard_approx)
#     assert espnet_crossentropy_loss.forward(tensor3, tensor4).data.numpy() == pytest.approx(0.57556236, standard_approx)
