"""Import smoke test: errors/validators are importable standalone (no circular import) (#614)."""

import subprocess
import sys


def test_errors_validators_import_standalone():
    code = "import academic_paper.errors, academic_paper.validators"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_server_reexports_same_objects():
    from academic_paper import errors, server, validators

    assert server._http_exc_for is errors._http_exc_for
    assert server._sanitize_text is validators._sanitize_text
    assert server._parse_list_field is validators._parse_list_field
    assert server._validate_published_date is validators._validate_published_date
