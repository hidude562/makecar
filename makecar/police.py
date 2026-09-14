"""Procedural police cars: marked patrol, unmarked patrol and true undercover.

`draw_features` rolls each piece of kit with the odds for the tier, and
`police_config` turns a roll into an ordinary makecar config (body style, paint,
livery, component assignments), so the car builds, edits and exports like any
other.  The drawn features are kept under the config's `police` key.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

TIERS = ("marked", "unmarked", "undercover")

# probability per tier: (marked, unmarked, undercover)
TABLE_ODDS: Dict[str, Tuple[float, float, float]] = {
    "concealed_lights": (0.80, 0.95, 0.30),   # LEDs in the grille, behind the windshield, on the rear deck, in the mirrors
    "siren": (1.00, 0.95, 0.30),
    "patrol_model": (0.95, 0.80, 0.25),       # a body style fleets actually buy: sedan, suv or pickup
    "fleet_paint": (0.85, 0.80, 0.75),        # black, white, gray or silver
    "dark_rear_glass": (0.80, 0.75, 0.50),
    "laptop_mount": (0.85, 0.65, 0.05),
    "steel_wheels": (0.60, 0.50, 0.15),
    "spotlight": (0.70, 0.40, 0.05),
    "extra_antennas": (0.60, 0.30, 0.05),
    "government_plates": (0.85, 0.30, 0.05),
    "push_bar": (0.60, 0.20, 0.05),
}
# the rest of a patrol car's kit (estimates)
EXTRA_ODDS: Dict[str, Tuple[float, float, float]] = {
    "light_bar": (0.90, 0.00, 0.00),
    "lettering": (1.00, 0.00, 0.00),
    "two_tone": (0.45, 0.00, 0.00),           # only when the paint is black or white
    "stripe": (0.70, 0.00, 0.00),
    "unit_number": (0.80, 0.00, 0.00),
    "partition": (0.85, 0.40, 0.05),
    "raised_stance": (0.50, 0.25, 0.05),
}
ODDS = {**TABLE_ODDS, **EXTRA_ODDS}

PATROL_STYLES = [("sedan", 0.55), ("suv", 0.35), ("pickup", 0.10)]
OTHER_STYLES = ["coupe", "hatchback", "wagon", "van", "sports"]
FLEET_PAINT = [("black", "#111214", 0.40), ("white", "#f4f4f0", 0.35), ("gray", "#6b6e73", 0.15), ("silver", "#b9bcc2", 0.10)]
OTHER_PAINT = [("dark blue", "#1c2a55"), ("dark green", "#1f3a2a"), ("maroon", "#5a1a22"), ("tan", "#b7a888"),
               ("red", "#a8221f"), ("brown", "#4d3a2a"), ("teal", "#2c6b6f"), ("orange", "#c8641b"), ("light blue", "#7f9fc4")]
AGENCIES = ["POLICE", "POLICE", "POLICE", "SHERIFF", "STATE TROOPER", "HIGHWAY PATROL", "STATE POLICE", "METRO POLICE"]
STRIPES = [("#1b3a8a", "#d9b13b"), ("#b3121b", "#f4f4f0"), ("#0d5c3a", "#d9b13b"), ("#14213d", "#c0c4cc")]
FONTS = [("sans-bold", 3.0), ("sans-condensed-bold", 2.0), ("serif-bold", 1.2), ("serif-condensed-bold", 1.0),
         ("mono-bold", 0.6), ("sans", 0.6)]
CITIES = ["SPRINGFIELD", "RIVERSIDE", "FAIRVIEW", "FRANKLIN", "GREENVILLE", "MADISON", "SALEM", "GEORGETOWN",
          "OAK RIDGE", "CLAYTON", "BRISTOL", "MILFORD", "KINGSTON", "ASHLAND", "CEDAR FALLS", "LAKEWOOD"]
UNITS = ["K-9 UNIT", "TRAFFIC UNIT", "PATROL", "COMMUNITY POLICING", "EMERGENCY 911", "SUPERVISOR", "DIAL 911"]
GOLD, WHITE, NAVY, BLACK = "#d9b13b", "#f4f4f0", "#14213d", "#0b0b0d"
DARK_GLASS = {"tint": "#0b0d10", "alpha": 0.88}


def _weighted(rng: random.Random, items: Sequence[tuple]):
    weights = [it[-1] for it in items]
    return rng.choices(items, weights=weights)[0][:-1]


def _luminance(color: str) -> float:
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def banner_style(rng: random.Random, light_paint: bool, stripe_color: str, stripe_accent: str) -> Dict[str, object]:
    """Lettering style for one car: font, slant, spacing, colours and an optional outline."""
    font = _weighted(rng, FONTS)[0]
    style: Dict[str, object] = {"font": font}
    if rng.random() < 0.3:
        style["slant_deg"] = round(rng.uniform(8, 14), 1)
    if rng.random() < 0.4:
        style["letter_spacing"] = round(rng.uniform(0.006, 0.025), 3)
    if font in ("serif-bold", "serif-condensed-bold", "sans") and rng.random() < 0.35:
        style["case"] = "title"
    fill = rng.choice([NAVY, stripe_color, BLACK]) if light_paint else rng.choice([WHITE, GOLD, stripe_accent])
    style["color"] = fill
    if rng.random() < 0.45:
        style["outline"] = round(rng.uniform(0.004, 0.009), 3)
        style["outline_color"] = rng.choice([WHITE, GOLD]) if _luminance(fill) < 0.5 else rng.choice([NAVY, stripe_color, BLACK])
    return style


def _title(text: str) -> str:
    words = text.split(" ")
    return " ".join(w if w in ("K-9", "911") else w.lower() if (w in ("OF", "THE") and i) else w.capitalize()
                    for i, w in enumerate(words))


def draw_features(tier: str, rng: random.Random) -> Dict[str, bool]:
    """Roll every feature independently with the tier's odds."""
    i = TIERS.index(tier)
    return {key: rng.random() < odds[i] for key, odds in ODDS.items()}


def describe(tier: str, features: Dict[str, bool], agency: str, paint_name: str, style: str) -> str:
    kit = [k.replace("_", " ") for k, v in features.items() if v and k not in ("patrol_model", "fleet_paint")]
    what = f"{tier} {agency.lower()} {style} in {paint_name}"
    return f"{what}; " + (", ".join(kit) if kit else "no visible kit")


def police_config(tier: str, seed: int = 0, name: Optional[str] = None, features: Optional[Dict[str, bool]] = None) -> dict:
    """A complete config dict for one police car.  Same tier + seed always gives the same car."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    rng = random.Random(f"police:{tier}:{seed}")
    f = dict(features) if features is not None else draw_features(tier, rng)
    for key in ODDS:
        f.setdefault(key, False)

    style = _weighted(rng, PATROL_STYLES)[0] if f["patrol_model"] else rng.choice(OTHER_STYLES)
    if f["fleet_paint"]:
        paint_name, paint = _weighted(rng, FLEET_PAINT)
    else:
        paint_name, paint = rng.choice(OTHER_PAINT)
    agency = rng.choice(AGENCIES)
    unit = str(rng.randrange(100, 1000))
    stripe_color, stripe_accent = rng.choice(STRIPES)

    # ------------------------------------------------------------- body
    two_tone = bool(f["two_tone"] and paint_name in ("black", "white"))
    livery: Dict[str, str] = {}
    secondary = "#f4f4f0" if paint_name != "white" else "#111214"
    if two_tone:
        if paint_name == "black":
            livery = {"doors": "secondary", "roof": "secondary"} if rng.random() < 0.7 else {"doors": "secondary"}
        else:
            livery = {"hood": "secondary", "roof": "secondary", "deck": "secondary"} if rng.random() < 0.5 else {"doors": "secondary"}
    f["two_tone"] = two_tone
    modifiers: Dict[str, float] = {}
    if f["raised_stance"]:
        modifiers["ground_clearance"] = round(rng.uniform(0.3, 0.6), 2)

    def text_color_on(zone: str) -> str:
        zone_paint = secondary if livery.get(zone) == "secondary" else paint
        return "#14213d" if _luminance(zone_paint) > 0.5 else "#f4f4f0"

    door_light = _luminance(secondary if livery.get("doors") == "secondary" else paint) > 0.5

    # ------------------------------------------------------------- components
    assign: Dict[str, object] = {"plate": {"options": {"region": "us"}}}
    if f["government_plates"]:
        assign["plate"] = {"options": {"style": "government", "text": f"PD {unit}"}}
    if f["light_bar"]:
        assign["roof_mount"] = {"component": "light.bar",
                                "options": {"colors": "red_blue" if rng.random() < 0.85 else "blue",
                                            "low_profile": rng.random() < 0.3}}
    if f["push_bar"]:
        assign["bumper_front"] = {"component": "bumper.push_bar", "options": {"siren": bool(f["siren"] and rng.random() < 0.5)}}
    if f["siren"]:
        assign["intake"] = {"options": {"siren": True}}
    if f["concealed_lights"]:
        spots = [k for k, p in (("grille", 0.8), ("visor", 0.6), ("deck", 0.5), ("mirrors", 0.35)) if rng.random() < p] or ["grille"]
        if "grille" in spots:
            assign["grille"] = {"options": {"emergency_lights": True}}
        if "visor" in spots:
            assign["headliner"] = {"options": {"visor_lights": True}}
        if "deck" in spots:
            assign["parcel_shelf"] = {"options": {"deck_lights": True}}
        if "mirrors" in spots:
            assign["mirror"] = {"options": {"emergency_light": True}}
        f["concealed_light_spots"] = spots  # type: ignore[assignment]
    if f["dark_rear_glass"]:
        for sel in ("glass_rear", "glass_quarter", "rear_window"):
            assign[sel] = {"options": dict(DARK_GLASS)}
    if f["laptop_mount"]:
        assign["console"] = {"options": {"laptop_mount": True}}
    if f["steel_wheels"]:
        assign["wheel"] = {"component": "wheel.steel", "options": {"hubcap": False, "rim_color": "#141516"}}
    if f["spotlight"]:
        assign["spotlight_L"] = "light.spotlight"
        if rng.random() < 0.35:
            assign["spotlight_R"] = "light.spotlight"
    if f["extra_antennas"]:
        assign["antenna_aux_L"] = "antenna.whip"
        if rng.random() < 0.5:
            assign["antenna_aux_R"] = {"component": "antenna.whip", "options": {"length": 0.55}}
    if f["partition"]:
        assign["cabin_partition"] = {"component": "partition.cage", "options": {"style": "bars" if rng.random() < 0.6 else "clear"}}

    stripe = bool(f["stripe"])
    stripe_opts = {"stripe": True, "stripe_color": stripe_color, "stripe_accent": stripe_accent} if stripe else {}
    lettering = banner_style(rng, door_light, stripe_color, stripe_accent)
    font = str(lettering["font"])
    title = lettering.get("case") == "title"
    word = _title(agency) if title else agency
    city = rng.choice(CITIES)
    county = f"{city} COUNTY" if agency == "SHERIFF" else f"CITY OF {city}"
    small = {k: v for k, v in lettering.items() if k not in ("case",)}
    small["font"] = "sans-condensed-bold" if font == "mono-bold" else font

    def text_block(text: str, size: float, y: float, **more) -> Dict[str, object]:
        return {"text": text, "size": size, "y": y, "font": font, **{k: v for k, v in lettering.items() if k != "case" and k != "font"}, **more}

    if f["lettering"]:
        stacked = rng.random() < 0.4
        door: Dict[str, object] = {"component": "decal.panel", "options": {**stripe_opts}}
        if stacked:
            door["options"]["texts"] = [
                {**text_block(_title(county) if title else county, round(rng.uniform(0.045, 0.06), 3), 0.54), "outline": 0.0,
                 "letter_spacing": 0.004},
                text_block(word, round(rng.uniform(0.11, 0.14), 3), 0.34),
            ]
        else:
            door["options"]["texts"] = [text_block(word, round(rng.uniform(0.12, 0.17), 3), round(rng.uniform(0.36, 0.46), 2))]
        assign["panel_door_front"] = door
        rear_text = rng.choice(UNITS) if rng.random() < 0.45 else (county if not stacked and rng.random() < 0.5 else None)
        if rear_text or stripe:
            rear: Dict[str, object] = {"component": "decal.panel", "options": {**stripe_opts}}
            if rear_text:
                rear["options"]["texts"] = [{**text_block(_title(rear_text) if title else rear_text, round(rng.uniform(0.05, 0.07), 3), 0.3),
                                             "font": small["font"], "outline": 0.0}]
            assign["panel_door_rear"] = rear
        assign["panel_deck"] = {"component": "decal.panel", "options": {"texts": [
            {**text_block(word, 0.10, 0.5), "color": text_color_on("deck"), "outline_color": lettering.get("outline_color", NAVY)}]}}
        if rng.random() < 0.35:
            assign["panel_hood"] = {"component": "decal.panel", "options": {"texts": [
                {**text_block(word, 0.16, 0.5), "x": 0.55, "color": text_color_on("hood"), "outline_color": lettering.get("outline_color", NAVY)}]}}
    if f["unit_number"]:
        assign["panel_quarter"] = {"component": "decal.panel", "options": {**stripe_opts, "texts": [
            {**text_block(unit, 0.09, 0.3), "font": small["font"], "case": None}]}}
        assign["panel_roof"] = {"component": "decal.panel", "options": {"texts": [
            {**text_block(unit, 0.32, 0.5), "x": 0.35, "font": small["font"], "color": text_color_on("roof"), "outline": 0.0}]}}
    elif stripe:
        assign["panel_quarter"] = {"component": "decal.panel", "options": dict(stripe_opts)}
    if stripe:
        assign["panel_fender"] = {"component": "decal.panel", "options": dict(stripe_opts)}

    hints = {"door_count": 4} if style in ("sedan", "suv", "pickup", "wagon", "hatchback", "van") else {}
    return {
        "name": name or f"police_{tier}_{seed}",
        "description": describe(tier, f, agency, paint_name, style),
        "seed": seed,
        "body": {"style": style, "modifiers": modifiers, "hints": hints, "livery": livery, "random": {"amount": 0.12}},
        "palette": {"paint": paint, "paint_secondary": secondary, "interior": "#1e1f22", "seat": "#26272b",
                    "interior_accent": "#3a3c40", "wood": "#2b2c30"},
        "components": {"defaults": True, "assign": assign},
        "police": {"tier": tier, "agency": agency, "unit": unit, "paint": paint_name, "features": f, "lettering": lettering},
    }


def generate(count: int, tier: str = "mixed", seed: int = 0) -> List[dict]:
    """`count` config dicts; tier 'mixed' picks a tier per car."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        t = rng.choice(TIERS) if tier == "mixed" else tier
        out.append(police_config(t, rng.randrange(1 << 30), name=f"police_{t}_{i + 1:02d}"))
    return out
