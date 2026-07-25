"""X/Y/Z Plot axes for the Great Conditioning Modifier.

The axis callbacks fire before `before_process_batch`, so they write into a dict the script
reads (and clears) there.  Registration is best-effort: X/Y/Z Plot is a built-in script, but
a user can disable it, and losing the grid axes must not take the extension down with it.
"""

from modules import scripts

_registered = False


def _grid_module():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("scripts.xyz_grid", "xyz_grid.py") and hasattr(data, "module"):
            return data.module
    return None


def xyz_support(cache: dict, methods: list, targets: list):
    global _registered
    if _registered:
        return

    xyz_grid = _grid_module()
    if xyz_grid is None:
        return

    def apply_field(field):
        def _(p, x, xs):
            cache.update({field: x})

        return _

    xyz_grid.axis_options.extend(
        [
            xyz_grid.AxisOption("[GCM] Enable", str, apply_field("enable"), choices=xyz_grid.boolean_choice(reverse=True)),
            xyz_grid.AxisOption("[GCM] Method", str, apply_field("method"), choices=lambda: list(methods)),
            xyz_grid.AxisOption("[GCM] Strength", float, apply_field("strength")),
            xyz_grid.AxisOption("[GCM] Seed", int, apply_field("seed")),
            xyz_grid.AxisOption("[GCM] Target", str, apply_field("target"), choices=lambda: list(targets)),
        ]
    )

    _registered = True
