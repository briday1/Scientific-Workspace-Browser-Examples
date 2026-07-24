import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts import download_passive_acoustics
from sigvue_examples.passive_acoustics.workspace import create_workspace


FIXTURES = Path(__file__).parents[1] / ".test-tmp" / "noaa-audio"


class PassiveAcousticsTests(unittest.TestCase):
    def test_pinned_noaa_clips_are_small_and_checksummed(self):
        self.assertEqual(2, len(download_passive_acoustics.CLIPS))
        self.assertLess(
            sum(clip.size for clip in download_passive_acoustics.CLIPS),
            2_000_000,
        )
        self.assertTrue(
            all(len(clip.sha256) == 64 for clip in download_passive_acoustics.CLIPS)
        )

    @unittest.skipUnless((FIXTURES / "a.wav").is_file(), "NOAA fixtures not downloaded")
    def test_workspace_builds_waveform_progressive_spectrogram_and_psd(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def local_download(remote, target, **kwargs):
                target = Path(target)
                target.mkdir(parents=True, exist_ok=True)
                source = FIXTURES / ("a.wav" if "Acall" in remote.filename else "b.wav")
                destination = target / remote.filename
                shutil.copyfile(source, destination)
                return destination

            with patch.object(
                download_passive_acoustics,
                "download_file",
                local_download,
            ):
                download_passive_acoustics.download_passive_acoustics(root)
            workspace = create_workspace({"data_root": root})
            resources = workspace.discover_items()
            opened = workspace.open_item(resources[0].identifier)
            figures = [view.callback({}) for view in opened.page.views[:3]]

        self.assertEqual(2, len(resources))
        self.assertTrue(figures[0]._sigvue_viewport_heatmap)
        self.assertEqual("scatter", figures[1].data[0].type)
        self.assertEqual("scatter", figures[2].data[0].type)
        self.assertTrue(np.isfinite(np.asarray(figures[2].data[0].y)).all())


if __name__ == "__main__":
    unittest.main()
