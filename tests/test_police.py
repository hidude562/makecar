"""The police car generator: odds per tier, feature -> config mapping, and the CLI."""
from __future__ import annotations

import json
import random

import pytest

from makecar import cli
from makecar.config import CarConfig
from makecar.pipeline import build_car
from makecar.police import EXTRA_ODDS, ODDS, TABLE_ODDS, TIERS, draw_features, generate, police_config


@pytest.mark.parametrize("tier", TIERS)
def test_features_follow_the_odds_table(tier):
    n = 3000
    rng = random.Random(f"odds-{tier}")
    counts = {k: 0 for k in ODDS}
    for _ in range(n):
        for k, v in draw_features(tier, rng).items():
            counts[k] += v
    i = TIERS.index(tier)
    for key, odds in ODDS.items():
        assert abs(counts[key] / n - odds[i]) < 0.035, (tier, key, counts[key] / n)


def test_the_table_and_the_extras_are_separate():
    assert set(TABLE_ODDS) & set(EXTRA_ODDS) == set() and set(ODDS) == set(TABLE_ODDS) | set(EXTRA_ODDS)
    assert len(TABLE_ODDS) == 11
    for key, odds in EXTRA_ODDS.items():
        assert odds[0] >= odds[1] >= odds[2], key   # marked cars carry the most kit


def test_same_seed_same_car_and_every_config_validates():
    assert police_config("marked", 7) == police_config("marked", 7)
    assert police_config("marked", 7) != police_config("marked", 8)
    for tier in TIERS:
        for seed in range(15):
            raw = police_config(tier, seed)
            cfg = CarConfig.from_dict(raw)
            assert cfg.raw["police"]["tier"] == tier and raw["description"].startswith(tier)
            assert raw["name"] == f"police_{tier}_{seed}"
    with pytest.raises(ValueError):
        police_config("secret", 1)


def test_all_features_on_map_to_the_kit():
    raw = police_config("marked", 3, features={k: True for k in ODDS})
    assign = raw["components"]["assign"]
    assert assign["roof_mount"]["component"] == "light.bar"
    assert assign["bumper_front"]["component"] == "bumper.push_bar"
    assert assign["spotlight_L"] == "light.spotlight"
    assert assign["antenna_aux_L"] == "antenna.whip"
    assert assign["cabin_partition"]["component"] == "partition.cage"
    assert assign["console"]["options"]["laptop_mount"] is True
    assert assign["intake"]["options"]["siren"] is True
    assert assign["plate"]["options"]["style"] == "government"
    assert assign["wheel"]["component"] == "wheel.steel" and assign["wheel"]["options"]["hubcap"] is False
    assert assign["rear_window"]["options"]["alpha"] > 0.8
    assert assign["panel_door_front"]["component"] == "decal.panel"
    door_texts = assign["panel_door_front"]["options"]["texts"]
    assert door_texts[-1]["text"].upper() == raw["police"]["agency"] and door_texts[-1]["font"] == raw["police"]["lettering"]["font"]
    assert assign["panel_door_front"]["options"]["stripe"] is True
    assert assign["panel_quarter"]["options"]["texts"][0]["text"] == raw["police"]["unit"]
    assert assign["plate"]["options"]["text"] == f"PD {raw['police']['unit']}"
    assert raw["body"]["modifiers"]["ground_clearance"] > 0
    assert raw["body"]["style"] in ("sedan", "suv", "pickup")
    assert raw["police"]["paint"] in ("black", "white", "gray", "silver")
    spots = raw["police"]["features"]["concealed_light_spots"]
    assert spots and set(spots) <= {"grille", "visor", "deck", "mirrors"}


def test_all_features_off_is_a_plain_car():
    raw = police_config("undercover", 4, features={k: False for k in ODDS})
    assign = raw["components"]["assign"]
    assert set(assign) == {"plate"} and assign["plate"]["options"] == {"region": "us"}
    assert raw["body"]["livery"] == {} and raw["body"]["modifiers"] == {}
    assert raw["body"]["style"] in ("coupe", "hatchback", "wagon", "van", "sports")
    assert "no visible kit" in raw["description"]


def test_two_tone_needs_black_or_white_paint():
    black = next(police_config("marked", s, features={**{k: False for k in ODDS}, "fleet_paint": True, "two_tone": True})
                 for s in range(40) if police_config("marked", s, features={**{k: False for k in ODDS}, "fleet_paint": True, "two_tone": True})["police"]["paint"] == "black")
    assert black["body"]["livery"] and black["police"]["features"]["two_tone"] is True
    other = police_config("marked", 1, features={**{k: False for k in ODDS}, "two_tone": True})
    assert other["body"]["livery"] == {} and other["police"]["features"]["two_tone"] is False


def test_banners_vary_between_cars():
    seen = {"font": set(), "outline": 0, "slant": 0, "spacing": 0, "title": 0, "stacked": 0, "rear_text": 0}
    for seed in range(60):
        raw = police_config("marked", seed, features={k: True for k in ODDS})
        style = raw["police"]["lettering"]
        seen["font"].add(style["font"])
        seen["outline"] += bool(style.get("outline"))
        seen["slant"] += bool(style.get("slant_deg"))
        seen["spacing"] += bool(style.get("letter_spacing"))
        seen["title"] += style.get("case") == "title"
        texts = raw["components"]["assign"]["panel_door_front"]["options"]["texts"]
        seen["stacked"] += len(texts) == 2
        seen["rear_text"] += "texts" in raw["components"]["assign"]["panel_door_rear"]["options"]
        for block in texts:
            assert set(block) <= {"text", "size", "y", "x", "font", "color", "slant_deg", "letter_spacing", "outline", "outline_color", "case"}
    assert len(seen["font"]) >= 4
    for key in ("outline", "slant", "spacing", "title", "stacked", "rear_text"):
        assert 5 <= seen[key] <= 55, (key, seen[key])


def test_generate_mixes_tiers_and_names_cars():
    cars = generate(6, "mixed", seed=2)
    assert [c["name"][:7] for c in cars] == ["police_"] * 6
    assert len({c["police"]["tier"] for c in generate(30, "mixed", seed=2)}) == 3
    assert all(c["police"]["tier"] == "unmarked" for c in generate(4, "unmarked", seed=1))


@pytest.mark.parametrize("tier", TIERS)
def test_a_generated_car_builds(tier):
    features = {k: True for k in ODDS} if tier == "marked" else None
    raw = police_config(tier, 21, features=features)
    if tier == "marked":
        raw["body"]["style"] = "sedan"
    cfg = CarConfig.from_dict(raw)
    res, asm, _ = build_car(cfg)
    comps = {i.component for i in asm.instances}
    if tier == "marked":
        assert {"light.bar", "bumper.push_bar", "light.spotlight", "antenna.whip", "partition.cage", "mount.laptop",
                "decal.panel"} <= comps
        door = next(i for i in asm.instances if i.connector.name == "panel_door_front_L")
        assert door.result.info["texts"][-1]["text"].upper() == raw["police"]["agency"]
        plate = next(i for i in asm.instances if i.connector.name == "plate_front")
        assert "number/glyph_0" in plate.result.mesh.groups and "band/glyph_0" in plate.result.mesh.groups
        if raw["body"]["livery"]:
            assert "paint_secondary" in res.mesh.face_materials
    assert asm.body.mesh.n_faces > 1000 and all(i.result.mesh.n_faces > 0 for i in asm.instances)


def test_cli_police_writes_cars_and_configs(tmp_path):
    cli.main(["police", "-n", "1", "--tier", "unmarked", "--seed", "5", "-o", str(tmp_path / "out"), "--configs", str(tmp_path / "cfg")])
    car = next((tmp_path / "out").iterdir())
    assert (car / "assembly.json").exists() and (car / "config.yaml").exists()
    saved = CarConfig.load(car / "config.yaml")
    assert saved.raw["police"]["tier"] == "unmarked"
    assert list((tmp_path / "cfg").glob("*.yaml"))
    info = json.loads((car / "assembly.json").read_text())
    assert info["stats"]["components"] > 50
    cli.main(["police", "-n", "2", "--seed", "1", "--no-build", "--configs", str(tmp_path / "only")])
    assert len(list((tmp_path / "only").glob("*.yaml"))) == 2 and not (tmp_path / "output").exists()
