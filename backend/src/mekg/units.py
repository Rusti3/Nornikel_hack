from __future__ import annotations

import re
from functools import lru_cache

from pint import UnitRegistry

from .models import NormalizedUnit


UNIT_ALIASES = {
    "%": ("percent", "dimensionless", 1.0),
    "мас.%": ("percent", "mass_fraction", 1.0),
    "масс.%": ("percent", "mass_fraction", 1.0),
    "wt.%": ("percent", "mass_fraction", 1.0),
    "mg/l": ("mg/L", "concentration", 1.0),
    "мг/л": ("mg/L", "concentration", 1.0),
    "мг/дм3": ("mg/L", "concentration", 1.0),
    "мг/дм³": ("mg/L", "concentration", 1.0),
    "g/l": ("g/L", "concentration", 1.0),
    "г/л": ("g/L", "concentration", 1.0),
    "г/дм3": ("g/L", "concentration", 1.0),
    "г/дм³": ("g/L", "concentration", 1.0),
    "г/т": ("g/t", "mass_fraction", 1.0),
    "ppm": ("ppm", "mass_fraction", 1.0),
    "мм": ("mm", "length", 1.0),
    "mm": ("mm", "length", 1.0),
    "см": ("cm", "length", 1.0),
    "cm": ("cm", "length", 1.0),
    "мкм": ("um", "length", 1.0),
    "µm": ("um", "length", 1.0),
    "нм": ("nm", "length", 1.0),
    "м/с": ("m/s", "velocity", 1.0),
    "m/s": ("m/s", "velocity", 1.0),
    "°c": ("degC", "temperature", 1.0),
    "c": ("degC", "temperature", 1.0),
    "с": ("degC", "temperature", 1.0),
    "k": ("K", "temperature", 1.0),
    "а/м2": ("A/m^2", "current_density", 1.0),
    "а/м²": ("A/m^2", "current_density", 1.0),
    "a/m2": ("A/m^2", "current_density", 1.0),
    "a/m²": ("A/m^2", "current_density", 1.0),
    "т/сут": ("t/day", "mass_flow_rate", 1.0),
    "t/d": ("t/day", "mass_flow_rate", 1.0),
    "м3/ч": ("m^3/hour", "volume_flow_rate", 1.0),
    "м³/ч": ("m^3/hour", "volume_flow_rate", 1.0),
    "квт·ч/т": ("kWh/t", "energy_consumption", 1.0),
    "kwh/t": ("kWh/t", "energy_consumption", 1.0),
    "па": ("Pa", "pressure", 1.0),
    "кпа": ("kPa", "pressure", 1.0),
    "мпа": ("MPa", "pressure", 1.0),
    "bar": ("bar", "pressure", 1.0),
    "мин": ("minute", "time", 1.0),
    "ч": ("hour", "time", 1.0),
    "h": ("hour", "time", 1.0),
    "сут": ("day", "time", 1.0),
    "безразм.": ("dimensionless", "dimensionless", 1.0),
    "dimensionless": ("dimensionless", "dimensionless", 1.0),
}


def _key(value: str) -> str:
    value = value.strip().lower().replace("−", "-").replace("·", "·")
    return re.sub(r"\s+", "", value)


class UnitNormalizer:
    def __init__(self) -> None:
        self.registry = UnitRegistry(autoconvert_offset_to_baseunit=True)

    @lru_cache(maxsize=512)
    def normalize(self, unit: str) -> NormalizedUnit:
        original = unit.strip()
        if not original:
            return NormalizedUnit(
                original=original, symbol="", dimension="unknown", valid=False, error="empty unit"
            )
        alias = UNIT_ALIASES.get(_key(original))
        if alias:
            symbol, dimension, factor = alias
            return NormalizedUnit(original=original, symbol=symbol, dimension=dimension, factor_to_base=factor)
        try:
            parsed = self.registry.parse_units(original)
            dimensionality = str(parsed.dimensionality)
            return NormalizedUnit(original=original, symbol=f"{parsed:~}", dimension=dimensionality)
        except Exception as exc:
            return NormalizedUnit(
                original=original,
                symbol=original,
                dimension="unknown",
                valid=False,
                error=str(exc),
            )

    def convert_value(self, value: float | None, normalized: NormalizedUnit) -> float | None:
        if value is None:
            return None
        return float(value) * normalized.factor_to_base
