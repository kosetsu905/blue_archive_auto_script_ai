from pathlib import Path


def test_training_loader_does_not_reference_independent_splits():
    source = (Path(__file__).resolve().parents[1] / "src" / "baas_student_recognition_ai" / "data.py").read_text(encoding="utf-8")
    training_function = source.split("def load_training_portraits", 1)[1]
    assert "independent_v1" not in training_function
    assert "independent_v2" not in training_function
