"""Plotly construction and view layout for analyzed waterfall products."""

from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import Presentation, ViewContext, add_viewport_heatmap

from ..io.sigmf.capabilities import read_sigmf_annotations
from ..style import TEAL, heatmap_grid_color, style_figure
from .models import WaterfallProducts


COLORMAPS = ("Plasma", "Viridis", "Cividis", "Inferno", "Magma", "Turbo")


def present(products: WaterfallProducts, ui: ViewContext) -> None:
    colormap = ui.colormap(
        "colormap",
        label="Waterfall colormap",
        default="Plasma",
        options=COLORMAPS,
        group="Display",
    )
    finite = products.waterfall_dbfs[np.isfinite(products.waterfall_dbfs)]
    automatic = (
        float(np.floor(np.percentile(finite, 3))) if finite.size else -90.0,
        float(np.ceil(np.percentile(finite, 99.5))) if finite.size else -20.0,
    )
    zmin, zmax = ui.limits(
        "dbfs_limits",
        label="dBFS limits",
        default=automatic,
        minimum=-140.0,
        maximum=0.0,
        step=1.0,
        group="Display",
    )
    spectrum_style = ui.trace_style(
        "spectrum_style",
        label="Average spectrum",
        color=TEAL,
        width=1.4,
        group="Display",
    )
    show_colorbar = ui.toggle(
        "show_colorbar",
        label="Show colorbar",
        default=True,
        group="Display",
    )
    show_annotations = ui.toggle(
        "show_annotations",
        label="Show annotations",
        default=True,
        group="Annotations",
    )
    annotation_color = ui.color(
        "annotation_region_color",
        label="Annotation color",
        default="#ffffff",
        group="Annotations",
    )
    annotation_width = float(ui.number(
        "annotation_region_width",
        label="Line weight",
        default=1.5,
        minimum=0.5,
        maximum=8.0,
        step=0.5,
        group="Annotations",
    ))
    annotation_opacity = float(ui.number(
        "annotation_region_opacity",
        label="Opacity",
        default=0.8,
        minimum=0.05,
        maximum=1.0,
        step=0.05,
        group="Annotations",
    ))
    with ui.details_group("Raster rendering"):
        render_width = int(ui.select(
            "render_width",
            label="Heatmap render width",
            default=1024,
            options=(256, 512, 1024, 2048),
        ))
        render_height = int(ui.select(
            "render_height",
            label="Heatmap render height",
            default=512,
            options=(128, 256, 512, 1024),
        ))
        aggregation = str(ui.select(
            "render_aggregation",
            label="Heatmap aggregation",
            default="mean",
            options=("max", "mean", "median"),
        ))
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.12, 0.88),
        vertical_spacing=0.04,
    )
    figure.add_trace(go.Scatter(
        x=products.frequency_mhz,
        y=products.spectrum_dbfs,
        mode=spectrum_style.mode,
        line=spectrum_style.line,
        marker=spectrum_style.plotly_marker,
        name="Average spectrum",
    ), row=1, col=1)
    add_viewport_heatmap(
        figure,
        viewport=ui.plot_viewport("lte-waterfall"),
        x=products.frequency_mhz,
        y=products.time_edges_ms,
        z=products.waterfall_dbfs,
        zmin=zmin,
        zmax=zmax,
        colorscale=colormap,
        showscale=show_colorbar,
        colorbar={"title": "dBFS"},
        render_width=render_width,
        render_height=render_height,
        aggregation=aggregation,
        row=2,
        col=1,
    )
    annotation_x: list[float | None] = []
    annotation_y: list[float | None] = []
    annotation_hover_x: list[float] = []
    annotation_hover_y: list[float] = []
    annotation_hover_text: list[str] = []
    for annotation in read_sigmf_annotations(products.recording) if show_annotations else ():
        y0 = annotation.start_seconds * 1e3
        y1 = y0 + (annotation.duration_seconds or 0.0) * 1e3
        if y1 < products.time_edges_ms[0] or y0 > products.time_edges_ms[-1]:
            continue
        x0 = (
            annotation.frequency_lower_hz / 1e6
            if annotation.frequency_lower_hz is not None
            else float(products.frequency_mhz[0])
        )
        x1 = (
            annotation.frequency_upper_hz / 1e6
            if annotation.frequency_upper_hz is not None
            else float(products.frequency_mhz[-1])
        )
        if y1 > y0:
            annotation_x.extend((x0, x1, x1, x0, x0, None))
            annotation_y.extend((y0, y0, y1, y1, y0, None))
        else:
            annotation_x.extend((x0, x1, None))
            annotation_y.extend((y0, y0, None))
        annotation_hover_x.append((x0 + x1) / 2)
        annotation_hover_y.append((y0 + y1) / 2)
        stop_seconds = annotation.start_seconds + (annotation.duration_seconds or 0.0)
        frequency = (
            f"{annotation.frequency_lower_hz / 1e6:.9g}–{annotation.frequency_upper_hz / 1e6:.9g} MHz"
            if annotation.frequency_lower_hz is not None and annotation.frequency_upper_hz is not None
            else "Full displayed frequency span"
        )
        details = [
            f"<b>{escape(annotation.label or 'Annotation')}</b>",
            f"Time: {annotation.start_seconds:.9g}–{stop_seconds:.9g} s",
            f"Duration: {(annotation.duration_seconds or 0.0):.9g} s",
            f"Frequency: {escape(frequency)}",
        ]
        if annotation.comment:
            details.append(escape(annotation.comment))
        annotation_hover_text.append("<br>".join(details))
    if annotation_x:
        figure.add_trace(go.Scattergl(
            x=annotation_x,
            y=annotation_y,
            mode="lines",
            line={"color": annotation_color, "width": annotation_width},
            opacity=annotation_opacity,
            name="Annotations",
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)
        figure.add_trace(go.Scattergl(
            x=annotation_hover_x,
            y=annotation_hover_y,
            mode="markers",
            marker={
                "color": annotation_color,
                "size": max(8.0, annotation_width * 4),
                "opacity": max(0.15, min(0.45, annotation_opacity)),
                "symbol": "square-open",
            },
            text=annotation_hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Annotation details",
            showlegend=False,
        ), row=2, col=1)
    figure.update_yaxes(title_text="Power (dBFS)", range=[zmin, zmax], autorange=False, row=1, col=1)
    figure.update_yaxes(
        title_text="Recording time (ms)",
        range=[float(products.time_edges_ms[0]), float(products.time_edges_ms[-1])],
        autorange=False,
        row=2,
        col=1,
    )
    frequency_step = (
        float(abs(products.frequency_mhz[1] - products.frequency_mhz[0]))
        if products.frequency_mhz.size > 1 else 1.0
    )
    frequency_range = [
        float(products.frequency_mhz[0] - frequency_step / 2),
        float(products.frequency_mhz[-1] + frequency_step / 2),
    ]
    figure.update_xaxes(
        title_text="RF frequency (MHz)",
        range=frequency_range,
        autorange=False,
        row=2,
        col=1,
    )
    figure.update_layout(uirevision=f"lte-waterfall:{products.recording.metadata_path}")
    title = str(products.recording.metadata["global"].get("core:description", "Synthetic LTE"))
    ui.stat("Sample rate", f"{products.recording.sample_rate / 1e6:g} MS/s")
    ui.stat("Center frequency", f"{products.recording.center_frequency / 1e6:g} MHz")
    with ui.tab("Spectrum + waterfall"):
        styled = style_figure(figure, ui.theme, title)
        styled.update_xaxes(gridcolor=heatmap_grid_color(ui.theme), gridwidth=0.35, row=2, col=1)
        styled.update_yaxes(gridcolor=heatmap_grid_color(ui.theme), gridwidth=0.35, row=2, col=1)
        ui.plot(styled, key="lte-waterfall", axis_navigation="bounded")


class WaterfallPresentation(Presentation[WaterfallProducts]):
    """Framework presentation object for the waterfall views."""

    def present(self, products: WaterfallProducts, ui: ViewContext) -> None:
        present(products, ui)
