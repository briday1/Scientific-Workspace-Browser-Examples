"""Calibration configuration for the radar analysis function."""

from sigvue.plugin import ParameterContext

from .models import LfmAnalysisProducts, LfmInput, LfmSettings
from .processing import THERMAL_NOISE_DBM_HZ, process_lfm


def configure_lfm(data: LfmInput, ui: ParameterContext) -> LfmSettings:
    channels = tuple(f"Channel {channel + 1}" for channel in range(data.ota_counts.shape[0]))
    phase_reference = str(
        ui.select(
            "phase_reference",
            label="Phase reference",
            default="Channel 1",
            options=channels,
            group="Calibration parameters",
        )
    )
    amplitude_reference = str(
        ui.select(
            "amplitude_reference",
            label="Amplitude reference",
            default="Min",
            options=(*channels, "Min"),
            group="Calibration parameters",
        )
    )
    adc_bits = int(
        ui.number(
            "adc_bits",
            label="Number of ADC bits",
            default=data.adc_bits,
            minimum=2,
            maximum=32,
            step=1,
            group="Calibration parameters",
        )
    )
    reference_noise_psd_dbm_hz = float(
        ui.number(
            "reference_noise_psd_dbm_hz",
            label="Reference noise PSD (dBm/Hz)",
            default=THERMAL_NOISE_DBM_HZ,
            minimum=-220.0,
            maximum=-100.0,
            step=0.1,
            group="Calibration parameters",
        )
    )
    return LfmSettings(
        adc_bits=adc_bits,
        phase_reference=phase_reference,
        amplitude_reference=amplitude_reference,
        reference_noise_psd_dbm_hz=reference_noise_psd_dbm_hz,
    )


def analyze_lfm(
    data: LfmInput,
    settings: LfmSettings | None,
) -> LfmAnalysisProducts:
    """Run the domain processor after enforcing configured calibration."""
    if settings is None:
        raise RuntimeError("LFM analysis requires configured settings")
    return process_lfm(data, settings)


__all__ = [
    "LfmAnalysisProducts",
    "LfmSettings",
    "analyze_lfm",
    "configure_lfm",
]
