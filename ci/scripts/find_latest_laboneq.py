#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pip",
#     "packaging"
# ]
# ///

# ruff: noqa: T201 # print in scripts is fine
# ruff: noqa: E731 # lambda's in scripts is fine
# ruff: noqa: S101 # Inline asserts in scripts are fine
"""Find the latest pre-release of laboneq."""

import argparse
import subprocess
import sys
from typing import Literal
from urllib.parse import urlparse

from packaging.version import InvalidVersion, Version

PACKAGE_NAME = "laboneq"


def validate_url(s: str) -> None:
    """Raise ValueError if URL can not be parsed."""
    parsed = urlparse(s)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"{s} is not valid url.")


def parse_pip_index_versions_output(s: str) -> list[Version]:
    """Parse the output from `pip index versions` and return a list of versions."""
    lines = s.splitlines()
    comma_separated_versions = lines[1].split(":", maxsplit=1)[1]
    versions = []
    for version_string in comma_separated_versions.split(","):
        try:
            versions.append(Version(version_string))
        except InvalidVersion:  # noqa: PERF203
            print(f"Ignoring version with incompatible scheme: {version_string}")
    return versions


def get_package_versions(index_url: str) -> list[Version]:
    """Query available versions of a package from the specified index using pip.

    Args:
        index_url: URL of the package index

    Returns:
        List of version strings

    Raises:
        CalledProcessError
    """
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "pip",
                "index",
                "versions",
                PACKAGE_NAME,
                "--pre",
                "--index-url",
                index_url,
                "--require-virtualenv",
                "--isolated",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error querying package index: '{e.stderr}'", file=sys.stderr)
        raise
    else:
        return parse_pip_index_versions_output(result.stdout)


def filter_per_build_type(
    versions: list[Version], build_type: Literal["dev", "alpha", "beta"]
) -> list[Version]:
    """Filter `versions` and return a list that match the given `build_type`."""
    match build_type:
        case "dev":
            predicate = lambda v: v.is_devrelease
        case "alpha":
            predicate = lambda v: v.pre is not None and v.pre[0] == "a"
        case "beta":
            predicate = lambda v: v.pre is not None and v.pre[0] == "b"
    return [v for v in versions if predicate(v)]


def test_filter_per_build_type() -> None:
    """Versions are filtered according to their build type."""
    versions = [
        Version("26.1.0a0"),
        Version("26.1.0a1"),
        Version("26.0.0a6"),
        Version("25.7.0b1"),
        Version("25.7.0b0"),
        Version("25.6.0b5"),
        Version("26.7.3.dev0"),
        Version("26.7.3.dev5"),
        Version("26.7.2.dev123"),
    ]
    assert filter_per_build_type(versions, "alpha") == [
        Version("26.1.0a0"),
        Version("26.1.0a1"),
        Version("26.0.0a6"),
    ]
    assert filter_per_build_type(versions, "beta") == [
        Version("25.7.0b1"),
        Version("25.7.0b0"),
        Version("25.6.0b5"),
    ]
    assert filter_per_build_type(versions, "dev") == [
        Version("26.7.3.dev0"),
        Version("26.7.3.dev5"),
        Version("26.7.2.dev123"),
    ]


def test_parse_pip_index_versions() -> None:
    """`pip index versions` output is parsed correctly."""
    example_output = """\
laboneq (25.10.3)
Available versions: 26.7.3dev5, 26.7.3.dev0, 26.7.2dev123+4125, 26.1.0a1, 26.1.0alpha0, 26.0.0alpha6, 25.7.0beta1, 25.7.0b0, 25.6.0beta5

"""  # noqa: E501
    versions = parse_pip_index_versions_output(example_output)
    assert versions == [
        Version("26.7.3.dev5"),
        Version("26.7.3.dev0"),
        Version("26.7.2.dev123+4125"),
        Version("26.1.0a1"),
        Version("26.1.0a0"),
        Version("26.0.0a6"),
        Version("25.7.0b1"),
        Version("25.7.0b0"),
        Version("25.6.0b5"),
    ]


def main() -> None:
    """Parse CLI arguments and run the program."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-url",
        required=True,
        help="URL of the package index to search",
    )
    parser.add_argument(
        "--build-type",
        choices=["dev", "alpha", "beta"],
        help="Build type to search for (choices: dev, alpha, beta)",
        required=True,
    )

    args = parser.parse_args()

    validate_url(args.index_url)

    try:
        versions = get_package_versions(args.index_url)
    except subprocess.CalledProcessError:
        # Assume that the function printed information message.
        sys.exit(1)

    desired_versions = sorted(filter_per_build_type(versions, args.build_type))

    if not desired_versions:
        print(
            f"No versions found for build type {args.build_type} "
            "in the index {args.index_url}",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.stdout.write(str(desired_versions[-1]))


if __name__ == "__main__":
    test_parse_pip_index_versions()
    test_filter_per_build_type()
    main()
