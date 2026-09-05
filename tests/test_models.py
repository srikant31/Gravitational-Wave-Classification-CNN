from pywavecnn.models import build_cnn_model, get_early_stopping


def test_build_cnn_model_output_shape():
    model = build_cnn_model(img_size=(64, 64), num_classes=5)
    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape == (None, 5)


def test_get_early_stopping_defaults():
    callback = get_early_stopping()
    assert callback.monitor == "val_accuracy"
    assert callback.patience == 5
