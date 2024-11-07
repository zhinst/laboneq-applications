#!/usr/bin/env python
"""Utility to check for presence of the correct license boilerplate."""

# ruff: noqa
import argparse
import re
import sys
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

# Boilerplate consists of copyright, license identifier, and one or more empty
# lines.

SEARCH = "".join(
    (
        r"(?P<copyright>(\n|[^\n]+\n))",
        r"(?P<comment>(//|\#)) SPDX-License-Identifier: (?P<license>[^\n]+)(\n\n)?",
        r"(?P<newlines>\n*)",
    )
)

REPLACEMENT = "".join(
    (
        r"\g<copyright>",
        r"\g<comment> SPDX-License-Identifier: \g<license>",
        "\n\n",
        r"\g<newlines>",
    )
)

SEARCH_RE = re.compile(SEARCH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Root path to scan", nargs="+")
    parser.add_argument(
        "--fixup",
        default=False,
        action="store_true",
        help="Fix up all issues",
    )
    parser.add_argument(
        "--exclude",
        default=[],
        help="Exclude paths from the scan",
        nargs="+",
    )

    args = parser.parse_args()
    modification_count = 0
    for root_path in args.path:
        root_directory = Path(root_path)

        for path in root_directory.glob("**/*"):
            if path.suffix not in (".py", ".rs"):
                continue

            if any(
                path.resolve().is_relative_to(Path(exclude).resolve())
                for exclude in args.exclude
            ):
                continue

            comment_char = "#" if path.suffix == ".py" else "//"
            original_text = path.read_text(encoding="utf-8")
            (modified_text, count) = SEARCH_RE.subn(REPLACEMENT, original_text)

            if count == 0:
                print("\n!! License missing:", path)
                license_header = (
                    REPLACEMENT.replace(r"\g<comment>", comment_char)
                    .replace(
                        r"\g<copyright>",
                        f"{comment_char} Copyright {datetime.now().year} Zurich Instruments AG\n",
                    )
                    .replace(r"\g<license>", "Apache-2.0")
                    .replace("\n\\g<newlines>", "\n")
                )
                modified_text = license_header + modified_text.lstrip()

            # Cleanup excessive newlines at end of file.
            if modified_text[-1] == modified_text[-2] == modified_text[-3] == "\n":
                modified_text = modified_text[:-2]

            if modified_text == original_text:
                continue

            if args.fixup:
                path.write_text(modified_text)
                print("!! Modified:", path)
            else:
                print("!! Modification needed:", path)

            modification_count += 1

            sys.stdout.writelines(
                unified_diff(
                    original_text.splitlines(keepends=True),
                    modified_text.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=f"{path!s} (modified)",
                )
            )

    if modification_count != 0:
        if args.fixup:
            print(f"\n {modification_count} file(s) were fixed up.")
        else:
            print(f"\n {modification_count} file(s) require changes.")
        sys.exit(1)

    print("All good!")


if __name__ == "__main__":
    main()
