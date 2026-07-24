"""Pure Plotly figure builders for LFM radar analysis products."""

from colorsys import hsv_to_rgb
from dataclasses import dataclass, replace
from math import ceil, sqrt

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import Annotation, TraceStyle, add_viewport_heatmap

from ..plugins import add_time_frequency_annotation_regions
from ..style import ORANGE, heatmap_grid_color, style_plotly
from .models import Calibration, LfmInput, Products
from .processing import (
    R_OHMS,
    _averaged_psd,
    _db10,
    _single_psd,
)


@dataclass(frozen=True)
class ChannelGrid:
    """A compact, near-square row-major grid for a channel collection."""

    channel_count: int
    rows: int
    columns: int

    def position(self, channel_index: int) -> tuple[int, int]:
        if not 0 <= channel_index < self.channel_count:
            raise IndexError(
                f"Channel index {channel_index} is outside this grid"
            )
        row, column = divmod(channel_index, self.columns)
        return row + 1, column + 1


def channel_grid(channel_count: int) -> ChannelGrid:
    """Return a near-square grid for a channel collection."""
    if channel_count < 1:
        raise ValueError("A channel grid requires at least one channel")
    columns = ceil(sqrt(channel_count))
    rows = ceil(channel_count / columns)
    return ChannelGrid(channel_count, rows, columns)


def channel_colors(
    channel_count: int,
    *,
    saturation: float = 0.68,
    value: float = 0.78,
) -> tuple[str, ...]:
    """Return one distinct HSV color for every channel in a collection."""
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or channel_count < 1
    ):
        raise ValueError("Channel colors require at least one channel")
    colors = []
    for index in range(channel_count):
        red, green, blue = hsv_to_rgb(
            index / channel_count,
            saturation,
            value,
        )
        colors.append(
            f"#{round(red * 255):02x}"
            f"{round(green * 255):02x}"
            f"{round(blue * 255):02x}"
        )
    return tuple(colors)


def _downsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select evenly spaced render points without changing analysis arrays."""
    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if max_points is None or x_values.size <= max_points:
        return x_values, y_values
    indexes = np.linspace(0, x_values.size - 1, max_points, dtype=np.int64)
    return x_values[indexes], y_values[indexes]


def _legend_inside_top_temporal_plot(figure: go.Figure, theme: str) -> go.Figure:
    """Keep calibration legends inside the upper time-domain subplot."""
    dark = theme == "dark"
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(16,37,45,0.78)" if dark else "rgba(255,255,255,0.82)",
            "bordercolor": "#36515b" if dark else "#dce5e8",
            "borderwidth": 1,
        }
    )
    return figure

def _phase_figure(
    counts: np.ndarray,
    calibration: Calibration,
    sample_rate: float,
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        row_heights=(0.42, 0.58),
        subplot_titles=("Amplitude", "Phase before", "Phase aligned"),
    )
    subset = counts[:, :512]
    time_us = np.arange(subset.shape[1]) / sample_rate * 1e6
    if max_points is not None and subset.shape[1] > max_points:
        indexes = np.linspace(0, subset.shape[1] - 1, max_points, dtype=np.int64)
        subset = subset[:, indexes]
        time_us = time_us[indexes]
    aligned = subset * np.exp(-1j * calibration.phase_offsets)[:, None]
    colors = channel_colors(subset.shape[0])
    for channel in range(subset.shape[0]):
        name = f"Channel {channel + 1}"
        line = {"color": colors[channel]}
        figure.add_trace(go.Scatter(x=time_us, y=np.abs(subset[channel]), name=name, line=line), row=1, col=1)
        figure.add_trace(go.Scatter(x=time_us, y=np.unwrap(np.angle(subset[channel])), name=name, line=line, showlegend=False), row=2, col=1)
        figure.add_trace(go.Scatter(x=time_us, y=np.unwrap(np.angle(aligned[channel])), name=name, line=line, showlegend=False), row=2, col=2)
    figure.update_xaxes(title_text="Time (us)")
    return _legend_inside_top_temporal_plot(
        style_plotly(
            figure,
            title=f"Phase calibration · reference Channel {calibration.phase_reference_channel + 1}",
            theme=theme,
            boxed_axes=True,
        ),
        theme,
    )

def _amplitude_figure(
    channels: np.ndarray,
    data: LfmInput,
    calibration: Calibration,
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Signal power", "Signal PSD"),
    )
    subset = channels[:, : min(4096, channels.shape[1])]
    time_us = np.arange(subset.shape[1]) / data.sample_rate * 1e6
    colors = channel_colors(subset.shape[0])
    for channel in range(subset.shape[0]):
        power = _db10((np.abs(subset[channel]) ** 2 / (2 * R_OHMS)) / 1e-3)
        frequency, psd = _single_psd(subset[channel], data.sample_rate)
        plot_time, plot_power = _downsample_xy(time_us, power, max_points)
        plot_frequency, plot_psd = _downsample_xy(frequency, psd, max_points)
        line = {"color": colors[channel]}
        figure.add_trace(go.Scatter(x=plot_time, y=plot_power, name=f"Channel {channel + 1}", line=line), row=1, col=1)
        figure.add_trace(go.Scatter(x=plot_frequency, y=plot_psd, name=f"Channel {channel + 1}", line=line, showlegend=False), row=2, col=1)
    figure.add_trace(go.Scatter(x=[time_us[0], time_us[-1]], y=[data.calibration_dbm] * 2, name="Incident power", line={"color": ORANGE, "dash": "dash"}), row=1, col=1)
    figure.update_xaxes(title_text="Time (us)", row=1, col=1)
    figure.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    return _legend_inside_top_temporal_plot(
        style_plotly(
            figure,
            title=f"Amplitude calibration · reference {calibration.amplitude_reference_label}",
            theme=theme,
            boxed_axes=True,
        ),
        theme,
    )

def _noise_figure(
    channels: np.ndarray,
    data: LfmInput,
    calibration: Calibration,
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Instantaneous noise power", "Averaged noise PSD"),
    )
    subset = channels[:, : min(4096, channels.shape[1])]
    time_us = np.arange(subset.shape[1]) / data.sample_rate * 1e6
    colors = channel_colors(subset.shape[0])
    for channel in range(subset.shape[0]):
        power = _db10((np.abs(subset[channel]) ** 2 / (2 * R_OHMS)) / 1e-3)
        frequency, psd = _averaged_psd(channels[channel], data.sample_rate)
        plot_time, plot_power = _downsample_xy(time_us, power, max_points)
        plot_frequency, plot_psd = _downsample_xy(frequency, psd, max_points)
        line = {"color": colors[channel]}
        figure.add_trace(go.Scatter(x=plot_time, y=plot_power, name=f"Channel {channel + 1}", line=line), row=1, col=1)
        figure.add_trace(go.Scatter(x=plot_frequency, y=plot_psd, name=f"Channel {channel + 1}", line=line, showlegend=False), row=2, col=1)
    for channel in range(subset.shape[0]):
        figure.add_trace(go.Scatter(x=[-data.sample_rate / 2, data.sample_rate / 2], y=[calibration.noise_psd_dbm_hz[channel]] * 2, name=f"Ch {channel + 1} measured floor", line={"dash": "dot"}), row=2, col=1)
    figure.update_xaxes(title_text="Time (us)", row=1, col=1)
    figure.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    return _legend_inside_top_temporal_plot(
        style_plotly(figure, title="Terminated-noise calibration", theme=theme, boxed_axes=True),
        theme,
    )

def _waterfall_figure(
    products: Products,
    domain: str,
    theme: str,
    colormap: str,
    zlimits: tuple[float, float],
    *,
    annotations: tuple[Annotation, ...] = (),
    window_start_seconds: float = 0.0,
    annotation_style: TraceStyle | None = None,
    show_annotations: bool = True,
    selected_channel: int | None = None,
    render_width: int = 1024,
    render_height: int = 512,
    render_aggregation: str = "mean",
    viewport: dict[str, object] | None = None,
) -> go.Figure:
    channel_count = products.time_waterfall_dbm.shape[0]
    channels = tuple(range(channel_count)) if selected_channel is None else (selected_channel,)
    tiled = selected_channel is None
    grid = channel_grid(channel_count if tiled else 1)
    figure = make_subplots(
        rows=grid.rows,
        cols=grid.columns,
        shared_xaxes="all" if tiled else False,
        shared_yaxes="all" if tiled else False,
        subplot_titles=[f"Channel {channel + 1}" for channel in channels],
    )
    for display_index, channel in enumerate(channels):
        row, col = grid.position(display_index)
        if domain == "time":
            x, z, title = products.fast_time_us, products.time_waterfall_dbm[channel], "Power (dBm)"
        else:
            x, z, title = products.frequencies_hz, products.psd_waterfall_dbm_hz[channel], "PSD (dBm/Hz)"
        add_viewport_heatmap(
            figure,
            viewport=viewport,
            x=x,
            y=products.slow_time_edges_s,
            z=z,
            zmin=zlimits[0],
            zmax=zlimits[1],
            colorscale=colormap,
            showscale=display_index == len(channels) - 1,
            colorbar={"title": title},
            render_width=render_width,
            render_height=render_height,
            aggregation=render_aggregation,
            row=row,
            col=col,
        )
        # Heatmap-only subplots do not advertise Plotly's box-select tool on
        # every browser. Two invisible selectable points enable rectangular
        # range selection without changing the rendered waterfall or its data.
        figure.add_trace(
            go.Scatter(
                x=(float(x[0]), float(x[-1])),
                y=(float(products.slow_time_edges_s[0]), float(products.slow_time_edges_s[-1])),
                mode="markers",
                marker={"opacity": 0.0, "size": 1},
                hoverinfo="skip",
                showlegend=False,
                name="Selection surface",
            ),
            row=row,
            col=col,
        )
    slow_time_edges = np.asarray(products.slow_time_edges_s, dtype=float)
    slow_time_start = float(slow_time_edges[0])
    slow_time_stop = float(slow_time_edges[-1])
    displayed_x = products.fast_time_us if domain == "time" else products.frequencies_hz
    if displayed_x.size > 1:
        x_spacing = float(np.median(np.diff(np.sort(np.asarray(displayed_x, dtype=float)))))
        x_start = float(np.min(displayed_x)) - x_spacing / 2
        x_stop = float(np.max(displayed_x)) + x_spacing / 2
    else:
        x_start = float(displayed_x[0]) - 0.5
        x_stop = float(displayed_x[0]) + 0.5
    if (
        domain == "frequency"
        and show_annotations
        and annotation_style is not None
        and products.frequencies_hz.size > 1
    ):
        view_lower_hz = float(np.min(products.frequencies_hz))
        view_upper_hz = float(np.max(products.frequencies_hz))
        view_stop_seconds = window_start_seconds + slow_time_stop
        relative_annotations = tuple(
            replace(
                annotation,
                start_seconds=(
                    annotation.start_seconds - window_start_seconds
                ),
                duration_seconds=(
                    view_stop_seconds - annotation.start_seconds
                    if annotation.duration_seconds is None
                    else annotation.duration_seconds
                ),
            )
            for annotation in annotations
        )
        for display_index, channel in enumerate(channels):
            row, col = grid.position(display_index)
            add_time_frequency_annotation_regions(
                figure,
                relative_annotations,
                time_range=(slow_time_start, slow_time_stop),
                frequency_range=(view_lower_hz, view_upper_hz),
                color=annotation_style.color,
                width=annotation_style.width,
                opacity=annotation_style.opacity,
                row=row,
                col=col,
            )
    figure.update_yaxes(
        title_text="Relative slow time (s)",
        range=[slow_time_start, slow_time_stop],
        autorange=False,
        uirevision=f"radar-waterfall-time:{slow_time_start:.12g}:{slow_time_stop:.12g}",
        col=1 if tiled else None,
    )
    if tiled:
        for column in range(2, grid.columns + 1):
            figure.update_yaxes(
                range=[slow_time_start, slow_time_stop],
                autorange=False,
                uirevision=f"radar-waterfall-time:{slow_time_start:.12g}:{slow_time_stop:.12g}",
                col=column,
            )
    figure.update_xaxes(
        range=[x_start, x_stop],
        autorange=False,
        uirevision=f"radar-waterfall-{domain}:{x_start:.12g}:{x_stop:.12g}",
    )
    figure.update_xaxes(
        title_text="Fast time (us)" if domain == "time" else "Frequency (Hz)",
        row=grid.rows,
    )
    styled = style_plotly(
        figure,
        title=("Fast-time power waterfall" if domain == "time" else "Frequency PSD waterfall")
        + ("" if tiled else f" · Channel {selected_channel + 1}"),
        theme=theme,
        boxed_axes=True,
    )
    styled.update_xaxes(gridcolor=heatmap_grid_color(theme), gridwidth=0.35)
    styled.update_yaxes(gridcolor=heatmap_grid_color(theme), gridwidth=0.35)
    return styled

def _time_figure(
    products: Products,
    calibration: Calibration,
    trace_styles: dict[str, TraceStyle],
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    channel_count = products.time_mean_dbm.shape[0]
    grid = channel_grid(channel_count)
    figure = make_subplots(
        rows=grid.rows,
        cols=grid.columns,
        shared_xaxes="all",
        subplot_titles=[f"Channel {channel + 1}" for channel in range(channel_count)],
    )
    for channel in range(channel_count):
        row, col = grid.position(channel)
        x = products.fast_time_us
        traces = (
            (products.time_mean_dbm[channel], "Mean", trace_styles["mean"]),
            (products.time_max_dbm[channel], "Max hold", trace_styles["max"]),
            (np.full(x.size, calibration.noise_power_dbm[channel]), "Noise power", trace_styles["noise"]),
            (np.full(x.size, calibration.full_scale_dbm[channel]), "Full scale", trace_styles["full_scale"]),
        )
        for y, name, trace_style in traces:
            plot_x, plot_y = _downsample_xy(x, y, max_points)
            figure.add_trace(
                go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    name=name,
                    mode=trace_style.mode,
                    line=trace_style.line,
                    marker=trace_style.plotly_marker,
                    showlegend=channel == 0,
                ),
                row=row,
                col=col,
            )
    figure.update_xaxes(title_text="Fast time (us)", row=grid.rows)
    figure.update_yaxes(title_text="Power (dBm)", col=1)
    return style_plotly(figure, title="Fast-time mean and max hold", theme=theme, boxed_axes=True)

def _combined_time_figure(
    products: Products,
    calibration: Calibration,
    aggregation: str,
    trace_styles: dict[str, TraceStyle],
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    values = products.time_max_dbm if aggregation == "max" else products.time_mean_dbm
    label = "Max hold" if aggregation == "max" else "Mean"
    figure = _combined_channel_figure(
        products.fast_time_us,
        values,
        _linear_average_db(calibration.noise_power_dbm),
        float(calibration.full_scale_dbm[calibration.amplitude_reference_channel]),
        label,
        "Average noise power",
        trace_styles[aggregation],
        trace_styles,
        max_points,
    )
    figure.update_xaxes(title_text="Fast time (us)")
    figure.update_yaxes(title_text="Power (dBm)")
    return style_plotly(figure, title=f"Combined fast-time {label.lower()}", theme=theme, boxed_axes=True)

def _frequency_figure(
    products: Products,
    calibration: Calibration,
    trace_styles: dict[str, TraceStyle],
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    channel_count = products.psd_mean_dbm_hz.shape[0]
    grid = channel_grid(channel_count)
    figure = make_subplots(
        rows=grid.rows,
        cols=grid.columns,
        shared_xaxes="all",
        subplot_titles=[f"Channel {channel + 1}" for channel in range(channel_count)],
    )
    for channel in range(channel_count):
        row, col = grid.position(channel)
        x = products.frequencies_hz
        traces = (
            (products.psd_mean_dbm_hz[channel], "Average", trace_styles["mean"]),
            (products.psd_max_dbm_hz[channel], "Max hold", trace_styles["max"]),
            (np.full(x.size, calibration.noise_psd_dbm_hz[channel]), "Noise PSD", trace_styles["noise"]),
            (np.full(x.size, calibration.full_scale_dbm[channel]), "Full scale", trace_styles["full_scale"]),
        )
        for y, name, trace_style in traces:
            plot_x, plot_y = _downsample_xy(x, y, max_points)
            figure.add_trace(
                go.Scatter(
                    x=plot_x,
                    y=plot_y,
                    name=name,
                    mode=trace_style.mode,
                    line=trace_style.line,
                    marker=trace_style.plotly_marker,
                    showlegend=channel == 0,
                ),
                row=row,
                col=col,
            )
    figure.update_xaxes(title_text="Frequency (Hz)", row=grid.rows)
    figure.update_yaxes(title_text="PSD (dBm/Hz)", col=1)
    return style_plotly(figure, title="Average and max-hold PSD", theme=theme, boxed_axes=True)

def _combined_frequency_figure(
    products: Products,
    calibration: Calibration,
    aggregation: str,
    trace_styles: dict[str, TraceStyle],
    theme: str,
    max_points: int | None = None,
) -> go.Figure:
    values = products.psd_max_dbm_hz if aggregation == "max" else products.psd_mean_dbm_hz
    label = "Max hold" if aggregation == "max" else "Mean"
    figure = _combined_channel_figure(
        products.frequencies_hz,
        values,
        _linear_average_db(calibration.noise_psd_dbm_hz),
        float(calibration.full_scale_dbm[calibration.amplitude_reference_channel]),
        label,
        "Average noise PSD",
        trace_styles[aggregation],
        trace_styles,
        max_points,
    )
    figure.update_xaxes(title_text="Frequency (Hz)")
    figure.update_yaxes(title_text="PSD (dBm/Hz)")
    return style_plotly(figure, title=f"Combined {label.lower()} PSD", theme=theme, boxed_axes=True)

def _combined_channel_figure(
    x: np.ndarray,
    values: np.ndarray,
    noise_value: float,
    full_scale_value: float,
    value_label: str,
    noise_label: str,
    value_style: TraceStyle,
    trace_styles: dict[str, TraceStyle],
    max_points: int | None = None,
) -> go.Figure:
    """Overlay channel results with shared post-calibration references."""
    figure = go.Figure()
    for channel, color in enumerate(channel_colors(values.shape[0])):
        channel_name = f"Channel {channel + 1}"
        plot_x, plot_y = _downsample_xy(x, values[channel], max_points)
        figure.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                name=f"{channel_name} {value_label}",
                mode=value_style.mode,
                line={**value_style.line, "color": value_style.color_with_opacity(color)},
                marker={**value_style.plotly_marker, "color": value_style.color_with_opacity(color)},
                legendgroup=channel_name,
            )
        )
    for reference, reference_label, reference_style in (
        (noise_value, noise_label, trace_styles["noise"]),
        (full_scale_value, "Full scale", trace_styles["full_scale"]),
    ):
        plot_x, plot_y = _downsample_xy(x, np.full(x.size, reference), max_points)
        figure.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                name=reference_label,
                mode="lines",
                line=reference_style.line,
            )
        )
    return figure

def _linear_average_db(values: np.ndarray) -> float:
    """Average power-like dB values in linear units before converting back to dB."""
    return float(_db10(np.mean(10 ** (np.asarray(values, dtype=float) / 10))))
