"""Plotly views for passive-acoustic recordings."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sigvue.plugin import add_viewport_heatmap

from ..style import TEAL, style_plotly
from .analysis import AcousticProducts


def waveform_figure(products: AcousticProducts, theme: str) -> go.Figure:
    stride = max(1, products.waveform.size // 12000)
    figure = go.Figure(go.Scatter(
        x=products.time_seconds[::stride],
        y=products.waveform[::stride],
        mode="lines",
        line={"color": TEAL, "width": 1},
        name="Hydrophone",
        hovertemplate="%{x:.3f} s<br>%{y:.4f} FS<extra></extra>",
    ))
    figure.update_xaxes(title_text="Recording time (s)")
    figure.update_yaxes(title_text="Amplitude (full scale)", range=[-1, 1])
    return style_plotly(figure, title="Hydrophone waveform", theme=theme, boxed_axes=True)


def spectrogram_figure(
    products: AcousticProducts,
    *,
    theme: str,
    viewport: dict[str, object] | None,
    render_width: int,
    render_height: int,
) -> go.Figure:
    values = products.spectrogram_dbfs
    finite = values[np.isfinite(values)]
    zmin = float(np.percentile(finite, 5))
    zmax = float(np.percentile(finite, 99.5))
    figure = go.Figure()
    figure.update_xaxes(
        title_text="Recording time (s)",
        range=[float(products.time_seconds[0]), float(products.time_seconds[-1])],
        autorange=False,
    )
    figure.update_yaxes(
        title_text="Acoustic frequency (kHz)",
        range=[0, float(products.frequency_khz[-1])],
        autorange=False,
    )
    add_viewport_heatmap(
        figure,
        viewport=viewport,
        render_width=render_width,
        render_height=render_height,
        aggregation="max",
        x=products.spectrogram_time_seconds,
        y=products.frequency_khz,
        z=values.T,
        zmin=zmin,
        zmax=zmax,
        colorscale="Portland",
        colorbar={"title": "dBFS"},
        hovertemplate="%{x:.3f} s<br>%{y:.2f} kHz<br>%{z:.1f} dBFS<extra></extra>",
    )
    return style_plotly(
        figure, title="Progressive acoustic spectrogram", theme=theme,
        boxed_axes=True,
    )


def psd_figure(products: AcousticProducts, theme: str) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=products.frequency_khz,
        y=products.psd_dbfs_hz,
        mode="lines",
        line={"color": TEAL, "width": 1.3},
        name="Mean PSD",
    ))
    figure.update_xaxes(
        title_text="Acoustic frequency (kHz)",
        range=[0, float(products.frequency_khz[-1])],
    )
    figure.update_yaxes(title_text="Mean power (dBFS/Hz)")
    return style_plotly(figure, title="Average acoustic spectrum", theme=theme, boxed_axes=True)
