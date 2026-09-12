#!/usr/bin/env python3
"""Inspection-only contact sheets; Pillow is not a makecar dependency."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def sheet(name, entries, columns=2, size=480):
    rows = (len(entries) + columns - 1) // columns
    image = Image.new('RGB', (columns * size, rows * (size + 30)), '#edf0f3')
    draw = ImageDraw.Draw(image)
    for i, (file, label) in enumerate(entries):
        x, y = (i % columns) * size, (i // columns) * (size + 30)
        with Image.open(ROOT / file) as source:
            image.paste(source.resize((size, size), Image.Resampling.LANCZOS), (x, y + 30))
        draw.text((x + 12, y + 8), label, fill='#18232e')
    image.save(ROOT / name)


if __name__ == '__main__':
    for index, families in enumerate((('mesh', 'five_spoke', 'twin_five'),
                                      ('multi_spoke', 'turbine', 'dish'), ('steel_cap',))):
        sheet(f'families_{index + 1}.png', [(f'after/{family}_{view}.png', f'{family} / {view}')
                                         for family in families for view in ('head_on', 'three_quarter')])
    sheet('tread_patterns.png', [(f'after/tread_{pattern}.png', pattern)
                                for pattern in ('directional_v', 'asymmetric_block', 'all_terrain', 'slick')])
    sheet('profiles.png', [(f'after/profile_{label}.png', label.replace('_', ' / '))
                           for label in ('245_35_R20', '205_65_R15')])
    sheet('brakes.png', [(f'after/brake_{pattern}_{view}.png', f'{pattern} / {view}')
                        for pattern in ('plain', 'drilled', 'slotted') for view in ('face', 'edge')])
    sheet('before_after.png', [(f'{stage}/default_{view}.png', f'{stage} / {view}')
                               for view in ('head_on', 'three_quarter', 'tread') for stage in ('before', 'after')])
