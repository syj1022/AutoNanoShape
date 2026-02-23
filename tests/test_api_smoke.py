from pathlib import Path


def test_api_exports():
    import autonanoshape as ans
    assert callable(ans.create_dataset)
    assert callable(ans.train_painn)
    assert callable(ans.predict)
