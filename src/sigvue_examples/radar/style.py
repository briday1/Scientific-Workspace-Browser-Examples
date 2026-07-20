"""Visual policies used only by the radar time/frequency pipeline."""

from colorsys import hsv_to_rgb


def hsv_channel_colors(
    count: int,
    *,
    saturation: float = 0.68,
    value: float = 0.78,
) -> tuple[str, ...]:
    """Return radar-channel colors sampled uniformly around the HSV hue wheel."""
    if count < 1:
        return ()
    colors = []
    for index in range(count):
        red, green, blue = hsv_to_rgb(index / count, saturation, value)
        colors.append(f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}")
    return tuple(colors)
