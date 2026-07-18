import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workspace_browser.web.application import create_app

from scientific_workspace_examples.sigmf import load_recording
from scripts.generate_segmented_results import EVENTS, generate as generate_segmented_results
from scripts.generate_minimal_sigmf import multiple_tones, qpsk, write_sigmf


ROOT = Path(__file__).resolve().parents[1]


class MinimalExampleTests(unittest.TestCase):
    def test_profile_loads_example_workspaces(self):
        app = create_app(config_path=ROOT / "browser.toml")
        self.assertEqual(
            [
                "qpsk-windowed",
                "acoustic-events-segmented",
                "multi-tone-seek",
                "lte-recordings",
                "lfm-live",
                "lfm-static",
            ],
            [workspace["id"] for workspace in app.list_workspaces()],
        )
        expected_mode_tags = {
            "qpsk-windowed": "windowed",
            "acoustic-events-segmented": "segmented",
            "multi-tone-seek": "seek",
            "lte-recordings": "windowed",
            "lfm-live": "live",
            "lfm-static": "static",
        }
        for workspace in app.list_workspaces():
            self.assertIn(expected_mode_tags[workspace["id"]], workspace["tags"])
            self.assertIn(expected_mode_tags[workspace["id"]], workspace["description"].lower())

    def test_compact_recordings_are_file_backed_and_use_distinct_modes(self):
        app = create_app(config_path=ROOT / "browser.toml")
        expected = {"qpsk-windowed": "windowed", "multi-tone-seek": "seek"}
        for workspace_id, expected_mode in expected.items():
            items = app.list_items(workspace_id, {})
            self.assertEqual(1, len(items))
            self.assertTrue(Path(items[0]["source_reference"]).is_file())
            page = app.open_item(workspace_id, items[0]["id"])["page"]
            self.assertEqual(expected_mode, page["playback"]["mode"])
            self.assertGreaterEqual(len(page["rendered_views"]), 1)

    def test_qpsk_window_has_received_power_overview(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("qpsk-windowed", {})[0]
        values = {"__window_start_seconds": "0.03", "__window_end_seconds": "0.04"}
        opened = app.open_item("qpsk-windowed", item["id"], values)
        playback = opened["page"]["playback"]
        self.assertEqual("Received power", playback["overview_label"])
        self.assertEqual(200, len(playback["overview_values"]))

    def test_qpsk_shows_constellation_then_eye(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("qpsk-windowed", {})[0]
        page = app.open_item("qpsk-windowed", item["id"])["page"]
        self.assertEqual(
            ["constellation", "eye"],
            [view["name"] for view in page["rendered_views"]],
        )
        constellation = page["rendered_views"][0]["value"]
        self.assertEqual([-0.75, 0.75], constellation["layout"]["xaxis"]["range"])
        self.assertEqual([-0.75, 0.75], constellation["layout"]["yaxis"]["range"])
        eye = page["rendered_views"][1]["value"]
        self.assertEqual([0, 2], eye["layout"]["xaxis"]["range"])
        self.assertEqual([-1, 1], eye["layout"]["yaxis"]["range"])

    def test_multi_tone_uses_one_plotly_figure_for_psd_and_waterfall(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("multi-tone-seek", {})[0]
        page = app.open_item("multi-tone-seek", item["id"])["page"]
        self.assertEqual(["tones"], [view["name"] for view in page["rendered_views"]])
        figure = page["rendered_views"][0]["value"]
        self.assertEqual(["scatter", "heatmap"], [trace["type"] for trace in figure["data"]])
        self.assertEqual([-80, 0], figure["layout"]["yaxis"]["range"])
        self.assertEqual([0, 0.25], figure["layout"]["yaxis2"]["range"])
        self.assertEqual("Buffer time (s)", figure["layout"]["yaxis2"]["title"]["text"])
        self.assertEqual([-50_000, 50_000], figure["layout"]["xaxis2"]["range"])
        self.assertEqual((-80, 0), (figure["data"][1]["zmin"], figure["data"][1]["zmax"]))

    def test_lte_recording_renders_rf_spectrum_and_waterfall(self):
        app = create_app(config_path=ROOT / "browser.toml")
        items = app.list_items("lte-recordings", {})
        self.assertEqual(2, len(items))
        item = next(item for item in items if "downlink" in item["id"])
        page = app.open_item("lte-recordings", item["id"])["page"]
        self.assertEqual("windowed", page["playback"]["mode"])
        self.assertEqual("Sliding median power (dBFS)", page["playback"]["overview_label"])
        self.assertEqual(400, len(page["playback"]["overview_values"]))
        self.assertEqual(["lte-spectrum"], [view["name"] for view in page["rendered_views"]])
        figure = page["rendered_views"][0]["value"]
        self.assertEqual(["scatter", "heatmap"], [trace["type"] for trace in figure["data"]])
        self.assertEqual("RF frequency (MHz)", figure["layout"]["xaxis2"]["title"]["text"])
        self.assertEqual("Recording time (ms)", figure["layout"]["yaxis2"]["title"]["text"])
        self.assertEqual("07.2f", figure["layout"]["xaxis2"]["tickformat"])
        self.assertEqual("07.2f", figure["layout"]["yaxis2"]["tickformat"])
        self.assertEqual([-90.0, -20.0], figure["layout"]["yaxis"]["range"])
        self.assertEqual(".1f", figure["layout"]["yaxis"]["tickformat"])
        self.assertEqual((-90.0, -20.0), (figure["data"][1]["zmin"], figure["data"][1]["zmax"]))
        self.assertEqual(".1f", figure["data"][1]["colorbar"]["tickformat"])
        self.assertEqual("#0d0887", figure["data"][1]["colorscale"][0][1])
        self.assertEqual(
            "lte-spectrum:LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta",
            figure["layout"]["uirevision"],
        )
        colormap = next(control for control in page["controls"] if control["name"] == "lte_colormap")
        self.assertEqual("colormap", colormap["control_type"])
        self.assertEqual(10, len(colormap["options"]))
        self.assertEqual(10, len(colormap["option_previews"]))
        self.assertEqual("Plasma", colormap["default"])
        self.assertEqual("details", colormap["placement"])
        self.assertEqual(
            ["lte_colormap", "lte_dbfs_limits"],
            [control["name"] for control in page["controls"] if control["group"] == "Spectrogram display"],
        )
        controls = {control["name"]: control for control in page["controls"]}
        limits = controls["lte_dbfs_limits"]
        self.assertEqual("limits", limits["control_type"])
        self.assertEqual((-90.0, -20.0), limits["default"])
        self.assertEqual((-120.0, 0.0, 1.0), (limits["minimum"], limits["maximum"], limits["step"]))
        self.assertEqual(4096, controls["lte_fft_size"]["default"])
        self.assertEqual("Hann", controls["lte_fft_window"]["default"])
        self.assertEqual(50, controls["lte_overlap_percent"]["default"])
        self.assertEqual(200, controls["lte_maximum_time_bins"]["default"])
        self.assertTrue(
            all(controls[name]["placement"] == "details" for name in (
                "lte_fft_size",
                "lte_fft_window",
                "lte_overlap_percent",
                "lte_maximum_time_bins",
            ))
        )
        self.assertEqual("806 MHz", page["statistics"]["Center frequency"])

        changed = app.open_item("lte-recordings", item["id"], {"lte_colormap": "Cividis"})["page"]
        changed_scale = changed["rendered_views"][0]["value"]["data"][1]["colorscale"]
        self.assertEqual("#00224e", changed_scale[0][1])

        changed = app.open_item("lte-recordings", item["id"], {"lte_dbfs_limits": "-82,-12"})["page"]
        changed_figure = changed["rendered_views"][0]["value"]
        self.assertEqual([-82.0, -12.0], changed_figure["layout"]["yaxis"]["range"])
        self.assertEqual((-82.0, -12.0), (changed_figure["data"][1]["zmin"], changed_figure["data"][1]["zmax"]))

        recording = load_recording(
            ROOT / "data/lte/downlink/LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta"
        )
        self.assertEqual("ci16_le", recording.datatype)
        self.assertEqual((1, 16), recording.read(0, 16).shape)
        self.assertLessEqual(float(abs(recording.read(0, 16)).max()), 1.0)

    def test_lte_uplink_uses_its_own_recording(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = next(item for item in app.list_items("lte-recordings", {}) if "uplink" in item["id"])
        self.assertIn("LTE_uplink_847MHz", item["id"])
        page = app.open_item("lte-recordings", item["id"])["page"]
        self.assertEqual("windowed", page["playback"]["mode"])
        self.assertEqual("847 MHz", page["statistics"]["Center frequency"])

    def test_segmented_acoustic_workspace_displays_irregular_stored_results(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("acoustic-events-segmented", {})[0]
        initial = app.open_item("acoustic-events-segmented", item["id"])["page"]
        self.assertEqual("segmented", initial["playback"]["mode"])
        starts = [segment["start_seconds"] for segment in initial["playback"]["segments"]]
        self.assertEqual([event[1] for event in EVENTS], starts)
        self.assertEqual(["waveform", "spectrum"], [view["name"] for view in initial["rendered_views"]])

        selected = app.open_item(
            "acoustic-events-segmented",
            item["id"],
            {"__segment_id": "event-005"},
        )["page"]
        self.assertEqual("event-005", selected["playback"]["selected_segment_id"])
        self.assertEqual("Valve actuation", selected["statistics"]["Stored event"])

    def test_sigmf_reader_loads_only_requested_frames(self):
        recording = load_recording(ROOT / "data/qpsk-windowed/qpsk.sigmf-meta")
        samples = recording.read(25, 100)
        self.assertEqual((1, 100), samples.shape)
        self.assertEqual((1, 10), recording.read(recording.sample_count - 10, 100).shape)

    def test_minimal_data_can_be_regenerated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(root, "qpsk", qpsk(duration=0.01), 100_000.0, "QPSK")
            write_sigmf(root, "multiple-tones", multiple_tones(duration=0.01), 100_000.0, "Tones")
            self.assertEqual(1_000, load_recording(root / "qpsk.sigmf-meta").sample_count)
            self.assertEqual(1_000, load_recording(root / "multiple-tones.sigmf-meta").sample_count)
            results = generate_segmented_results(root / "acoustic-events.json")
            self.assertTrue(results.is_file())


if __name__ == "__main__":
    unittest.main()
