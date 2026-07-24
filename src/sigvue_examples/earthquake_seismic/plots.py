"""Plotly views for earthquake seismology."""

import numpy as np
import plotly.graph_objects as go
from sigvue.plugin import add_viewport_heatmap

from ..style import ORANGE, TEAL, style_plotly
from .analysis import SeismicProducts


def waveform(products: SeismicProducts, theme: str) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=products.time_seconds,
        y=products.velocity_um_s,
        mode="lines",
        line={"color": TEAL, "width": 1},
        name="BHZ",
    ))
    figure.add_vline(x=89, line={"color": ORANGE, "dash": "dash"})
    figure.update_xaxes(title_text="Time since request start (s)")
    figure.update_yaxes(title_text="Vertical ground velocity (µm/s)")
    return style_plotly(figure, title="Anchorage earthquake waveform", theme=theme, boxed_axes=True)


def spectrogram(
    products: SeismicProducts,
    theme: str,
    viewport: dict[str, object] | None,
) -> go.Figure:
    finite = products.spectrogram_db[np.isfinite(products.spectrogram_db)]
    figure = go.Figure()
    figure.update_xaxes(
        title_text="Time since request start (s)",
        range=[float(products.time_seconds[0]), float(products.time_seconds[-1])],
    )
    figure.update_yaxes(
        title_text="Frequency (Hz)",
        range=[0, float(products.frequency_hz[-1])],
    )
    add_viewport_heatmap(
        figure,
        viewport=viewport,
        render_width=1024,
        render_height=512,
        aggregation="max",
        x=products.spectrogram_time_seconds,
        y=products.frequency_hz,
        z=products.spectrogram_db.T,
        zmin=float(np.percentile(finite, 5)),
        zmax=float(np.percentile(finite, 99.5)),
        colorscale="Portland",
        colorbar={"title": "d (m/s)²/Hz"},
    )
    return style_plotly(
        figure, title="Progressive seismic spectrogram", theme=theme,
        boxed_axes=True,
    )


def spectrum(products: SeismicProducts, theme: str) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=products.frequency_hz,
        y=products.psd_db,
        mode="lines",
        line={"color": TEAL},
        name="BHZ PSD",
    ))
    figure.update_xaxes(title_text="Frequency (Hz)")
    figure.update_yaxes(title_text="Power spectral density (dB re (m/s)²/Hz)")
    return style_plotly(figure, title="Average seismic spectrum", theme=theme, boxed_axes=True)
