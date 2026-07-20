"""Windowed ranged-read policy for one file or every collection member."""

from .domain import WaterfallWindow, WindowedWaterfallDelivery

__all__ = ["WaterfallWindow", "WindowedWaterfallDelivery"]
