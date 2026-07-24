"""Generate compact deterministic ci16 LTE-shaped fixtures for automated tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sigvue_examples.plugins.sigmf import write_sigmf_recording


RECORDINGS = (
    ("downlink", "LTE_downlink_806MHz_2022-04-09_30720ksps", 806_000_000.0, 20220409),
    ("uplink", "LTE_uplink_847MHz_2022-01-30_30720ksps", 847_000_000.0, 20220130),
)


def generate(root: Path) -> tuple[Path, ...]:
    sample_rate = 30_720_000.0
    count = 700_000
    time = np.arange(count) / sample_rate
    written = []
    for direction, name, center_hz, seed in RECORDINGS:
        rng = np.random.default_rng(seed)
        signal = 0.42 * np.exp(2j * np.pi * 2_100_000 * time)
        signal += 0.18 * np.exp(-2j * np.pi * 5_300_000 * time)
        signal += 0.025 * (rng.normal(size=count) + 1j * rng.normal(size=count))
        destination = root / direction
        metadata_path, data_path = write_sigmf_recording(
            destination,
            name,
            signal,
            sample_rate,
            datatype="ci16_le",
            description=(
                f"Compact deterministic LTE {direction} test fixture"
            ),
            captures=(
                {
                    "core:sample_start": 0,
                    "core:frequency": center_hz,
                },
            ),
        )
        written.extend((metadata_path, data_path))
    return tuple(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in generate(args.output):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
