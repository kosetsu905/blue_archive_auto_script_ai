from baas_student_recognition_ai.catalog import StudentCatalog
from baas_student_recognition_ai.config import load_roster


def test_roster_has_270_unique_students_and_all_aliases_resolve():
    rows = load_roster()
    assert len(rows) == 270
    catalog = StudentCatalog(rows)
    assert len(catalog.records) == 270
    for row in rows:
        for key in ("Global_name", "CN_name", "JP_name"):
            assert catalog.resolve(row[key]).canonical_name == row["Global_name"]
