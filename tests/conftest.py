import pytest
import yaml


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("test_data/test_espnet.yaml", marks=pytest.mark.espnet),
        pytest.param("test_data/test_unet_seresnext.yaml", marks=pytest.mark.seresnext),
        pytest.param("test_data/test_unet_resnet.yaml", marks=pytest.mark.resnet),
        pytest.param("test_data/test_unet_densenet.yaml", marks=pytest.mark.densenet),
    ],
)
def config(request):
    with open(request.param, "r") as f:
        config = yaml.safe_load(f)
    return config
