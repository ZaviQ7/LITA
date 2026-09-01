# LITA model directory

Place a specialized weapon detector at:

`models/lita_weapon.pt`

LITA will automatically load it at startup. Until then, the application falls
back to the stock COCO detector, which only provides knife/scissors coverage and
should not be treated as firearm detection.

Recommended custom labels:
- `handgun`
- `long_gun`
- `knife`
- `other_weapon`

Aliases for common labels are defined in `config.py`.
