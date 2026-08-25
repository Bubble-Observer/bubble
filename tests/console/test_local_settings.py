"""Contracts for the console-only, allowlisted local settings writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from leave_information_bubble.console.local_settings import update_local_settings


def test_local_settings_update_preserves_unrelated_lines_and_replaces_supported_keys(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# keep this comment\nUNRELATED=one\nDEEPSEEK_MODEL=old\nDEEPSEEK_MODEL=duplicate\n",
        encoding="utf-8",
    )

    update_local_settings(
        env_file,
        {"DEEPSEEK_MODEL": "deepseek-v4", "NGA_COOKIE": "a=b; c=d"},
    )

    assert env_file.read_text(encoding="utf-8") == (
        '# keep this comment\nUNRELATED=one\nDEEPSEEK_MODEL="deepseek-v4"\n'
        'NGA_COOKIE="a=b; c=d"\n'
    )
    assert list(tmp_path.iterdir()) == [env_file]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"UNSUPPORTED": "value"}, "unsupported local setting"),
        ({"DEEPSEEK_API_KEY": "first\nsecond"}, "single-line"),
        ({"DEEPSEEK_API_KEY": "prefix-${SHOULD_NOT_EXPAND}"}, "interpolation"),
    ],
)
def test_local_settings_update_rejects_unknown_or_multiline_values_without_writing(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    env_file = tmp_path / ".env"

    with pytest.raises(ValueError, match=message):
        update_local_settings(env_file, updates)

    assert not env_file.exists()
