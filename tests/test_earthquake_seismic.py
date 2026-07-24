from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import download_earthquake_seismic
from sigvue_examples.earthquake_seismic.workspace import create_workspace


FIXTURE = Path(__file__).parents[1] / ".test-tmp" / "cola.csv"


class EarthquakeSeismicTests(unittest.TestCase):
    def test_geocsv_parser_preserves_samples_and_calibration_metadata(self):
        text = "\n".join((
            "# dataset: GeoCSV 2.0",
            "# SID: IU_COLA_00_BHZ",
            "# sample_rate_hz: 40",
            "# scale_factor: 2",
            "# scale_units: m/s",
            "Time, Sample",
            "2018-11-30T17:28:00Z, 4",
            "2018-11-30T17:28:00.025Z, -2",
        ))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wave.csv"
            path.write_text(text, encoding="utf-8")
            samples, metadata = download_earthquake_seismic.read_geocsv(path)
        self.assertEqual([4, -2], samples.tolist())
        self.assertEqual("IU_COLA_00_BHZ", metadata["SID"])

    @unittest.skipUnless(FIXTURE.is_file(), "live EarthScope fixture unavailable")
    def test_live_fixture_builds_three_scientific_plots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                download_earthquake_seismic,
                "download_file",
                return_value=FIXTURE,
            ):
                download_earthquake_seismic.download_earthquake_seismic(root)
            workspace = create_workspace({"data_root": root})
            opened = workspace.open_item(workspace.discover_items()[0].identifier)
            figures = [view.callback({}) for view in opened.page.views[:3]]
        self.assertEqual(["scatter", "heatmap", "scatter"], [f.data[0].type for f in figures])
        self.assertTrue(figures[1]._sigvue_viewport_heatmap)


if __name__ == "__main__":
    unittest.main()
