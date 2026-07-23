"""UI orchestration for LFM radar analysis products."""

from sigvue.plugin import Presentation, ViewContext

from ..style import ORANGE, TEAL
from ..memory import format_bytes
from .domain import LfmAnalysisProducts
from .plots import (
    _amplitude_figure, _combined_frequency_figure, _combined_time_figure,
    _frequency_figure, _noise_figure, _phase_figure, _time_figure,
    _waterfall_figure,
)


COLORMAPS = ("Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Blues", "Greens", "Hot", "Jet")
TIME_WATERFALL_LIMITS_DBM = (-100.0, -10.0)
PSD_WATERFALL_LIMITS_DBM_HZ = (-180.0, -80.0)


def present_lfm(results: LfmAnalysisProducts, ui: ViewContext) -> None:
    data = results.data
    calibration = results.calibration
    products = results.signal
    trace_styles = {
        "mean": ui.trace_style("mean_trace", label="Mean / average", color=TEAL, width=1.5),
        "max": ui.trace_style("max_trace", label="Max hold", color=ORANGE, width=1.5),
        "noise": ui.trace_style("noise_trace", label="Noise reference", color="#8f9fa6", width=1.0, line_style="dot"),
        "full_scale": ui.trace_style("full_scale_trace", label="Full scale", color="#60717d", width=1.0, line_style="dash"),
    }
    show_annotations = ui.toggle(
        "lfm_show_annotations", default=True, label="Show annotations", group="Annotation display"
    )
    annotation_style = ui.trace_style(
        "lfm_annotation_region",
        label="Annotation boxes",
        color="#ffffff",
        width=0.5,
        opacity=0.6,
        line_style="solid",
        group="Annotation display",
    )
    waterfall_colormap = ui.colormap(
        "lfm_waterfall_colormap",
        label="Colormap",
        default="Plasma",
        options=COLORMAPS,
        group="Waterfall display",
    )
    time_waterfall_limits = ui.limits(
        "lfm_time_waterfall_limits",
        label="Fast-time power z-limits (dBm)",
        default=TIME_WATERFALL_LIMITS_DBM,
        minimum=-200.0,
        maximum=50.0,
        step=1.0,
        group="Waterfall display",
    )
    psd_waterfall_limits = ui.limits(
        "lfm_psd_waterfall_limits",
        label="Frequency PSD z-limits (dBm/Hz)",
        default=PSD_WATERFALL_LIMITS_DBM_HZ,
        minimum=-240.0,
        maximum=0.0,
        step=1.0,
        group="Waterfall display",
    )
    many_channels = products.time_waterfall_dbm.shape[0] > 8
    waterfall_render = {
        "render_width": int(ui.select(
            "lfm_waterfall_render_width",
            label="Heatmap render width",
            default=256 if many_channels else 1024,
            options=(256, 512, 1024, 2048),
            group="Raster rendering",
        )),
        "render_height": int(ui.select(
            "lfm_waterfall_render_height",
            label="Heatmap render height",
            default=128 if many_channels else 512,
            options=(128, 256, 512, 1024),
            group="Raster rendering",
        )),
        "render_aggregation": str(ui.select(
            "lfm_waterfall_render_aggregation",
            label="Heatmap aggregation",
            default="mean",
            options=("max", "mean", "median"),
            group="Raster rendering",
        )),
    }
    line_render_points = ui.render_points(
        "lfm_line_render_points",
        label="Maximum line points",
        default=1024 if many_channels else 2048,
        minimum=128,
        maximum=8192,
        step=128,
        group="Raster rendering",
    )

    with ui.tab("Waterfall"):
        waterfall_views = {}
        channel_choices = (
            *((f"Ch{channel + 1}", channel) for channel in range(products.time_waterfall_dbm.shape[0])),
            ("Multi", None),
        )
        for channel_index, (channel_label, channel) in enumerate(channel_choices):
            time_view_key = f"waterfall-domain-{2 * channel_index}"
            frequency_view_key = f"waterfall-domain-{2 * channel_index + 1}"
            waterfall_views[("Fast-time power", channel_label)] = (
                lambda selected=channel, view_key=time_view_key: _waterfall_figure(
                    products,
                    "time",
                    ui.theme,
                    waterfall_colormap,
                    time_waterfall_limits,
                    annotations=data.annotations,
                    window_start_seconds=data.start_sample / data.sample_rate,
                    annotation_style=annotation_style,
                    show_annotations=show_annotations,
                    selected_channel=selected,
                    viewport=ui.plot_viewport(view_key),
                    **waterfall_render,
                )
            )
            waterfall_views[("Frequency PSD", channel_label)] = (
                lambda selected=channel, view_key=frequency_view_key: _waterfall_figure(
                    products,
                    "frequency",
                    ui.theme,
                    waterfall_colormap,
                    psd_waterfall_limits,
                    annotations=data.annotations,
                    window_start_seconds=data.start_sample / data.sample_rate,
                    annotation_style=annotation_style,
                    show_annotations=show_annotations,
                    selected_channel=selected,
                    viewport=ui.plot_viewport(view_key),
                    **waterfall_render,
                )
            )
        ui.view_switcher(
            ("Domain", "Channel"),
            waterfall_views,
            key="waterfall-domain",
            selector=("buttons", "dropdown"),
            axis_navigation="bounded",
        )
    with ui.tab("Time Domain"):
        ui.view_switcher(
            "View",
            {
                "Multi": lambda: _time_figure(products, calibration, trace_styles, ui.theme, line_render_points),
                "Combined max": lambda: _combined_time_figure(products, calibration, "max", trace_styles, ui.theme, line_render_points),
                "Combined mean": lambda: _combined_time_figure(products, calibration, "mean", trace_styles, ui.theme, line_render_points),
            },
            key="time-view",
            selector="buttons",
        )
    with ui.tab("Frequency Domain"):
        ui.view_switcher(
            "View",
            {
                "Multi": lambda: _frequency_figure(products, calibration, trace_styles, ui.theme, line_render_points),
                "Combined max": lambda: _combined_frequency_figure(products, calibration, "max", trace_styles, ui.theme, line_render_points),
                "Combined mean": lambda: _combined_frequency_figure(products, calibration, "mean", trace_styles, ui.theme, line_render_points),
            },
            key="frequency-view",
            selector="buttons",
        )
    with ui.tab("Calibration", update="static"):
        with ui.switcher("Calibration view", key="calibration-view", selector="buttons"):
            with ui.switcher_view("Phase", columns=(0.24, 0.76)):
                with ui.group("column"):
                    ui.place_parameters("phase_reference", label="Calibration parameters")
                    ui.table(results.phase_rows, key="phase-diagnostics", depends_on=("phase_reference",))
                ui.plot(
                    lambda: _phase_figure(data.calibration_counts, calibration, data.sample_rate, ui.theme, line_render_points),
                    key="phase-plot",
                    depends_on=("phase_reference",),
                )
            with ui.switcher_view("Amplitude", columns=(0.3, 0.7)):
                with ui.group("column"):
                    ui.place_parameters(
                        "amplitude_reference",
                        "adc_bits",
                        label="Calibration parameters",
                    )
                    ui.text(
                        results.amplitude_summary,
                        key="amplitude-summary",
                        depends_on=("amplitude_reference", "adc_bits"),
                    )
                    ui.table(
                        results.amplitude_rows,
                        key="amplitude-diagnostics",
                        depends_on=("amplitude_reference", "adc_bits"),
                    )
                ui.plot(
                    lambda: _amplitude_figure(results.calibrated_tone, data, calibration, ui.theme, line_render_points),
                    key="amplitude-plot",
                    depends_on=("amplitude_reference", "adc_bits"),
                )
            with ui.switcher_view("Noise", columns=(0.3, 0.7)):
                with ui.group("column"):
                    ui.place_parameters("reference_noise_psd_dbm_hz", label="Calibration parameters")
                    ui.table(
                        results.noise_rows,
                        key="noise-diagnostics",
                        depends_on=("reference_noise_psd_dbm_hz",),
                    )
                ui.plot(
                    lambda: _noise_figure(results.calibrated_noise, data, calibration, ui.theme, line_render_points),
                    key="noise-plot",
                )

    input_arrays = (data.calibration_counts, data.noise_counts, data.ota_counts)
    resident_input_bytes = sum(array.nbytes for array in {id(value): value for value in input_arrays}.values())
    ui.stat("Channels delivered", f"{data.ota_counts.shape[0]:,}")
    ui.stat("Samples per channel", f"{data.ota_counts.shape[1]:,}")
    ui.stat("Complex samples delivered", f"{data.ota_counts.size:,}")
    ui.stat("Input buffer memory", format_bytes(resident_input_bytes))
    ui.stat("Duration delivered", f"{data.ota_counts.shape[1] / data.sample_rate:g} s")
    ui.stat("Processing PRI", f"{data.pri_samples / data.sample_rate:g} s")
    ui.stat("Sample rate", f"{data.sample_rate / 1e6:g} MHz")
    ui.stat("Waterfall rows", f"{products.slow_time_s.size:,}")
    ui.stat("Fast-time points", f"{products.fast_time_us.size:,}")
    ui.stat("Frequency points", f"{products.frequencies_hz.size:,}")


class LfmPresentation(Presentation[LfmAnalysisProducts]):
    """Framework presentation for calibrated LFM products."""

    def present(self, results: LfmAnalysisProducts, ui: ViewContext) -> None:
        present_lfm(results, ui)
