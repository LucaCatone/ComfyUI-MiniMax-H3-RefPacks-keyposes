"""H3 integration; install alongside the shared BExplains_RefPacks package."""
import importlib.util
from pathlib import Path
import sys

if "bexplains_refpacks" not in sys.modules:
    # Registry, source-clone and legacy installations use different folder names.
    roots = [Path(__file__).resolve().parent.parent]
    import folder_paths
    roots.extend(Path(root) for root in folder_paths.get_folder_paths("custom_nodes"))
    candidates = [root / name / "__init__.py" for root in dict.fromkeys(roots)
                  for name in ("bexplains-refpacks", "ComfyUI-RefPacks", "BExplains_RefPacks", "ComfyUI-RefPacks-keyposes")]
    shared = next((path for path in candidates if path.is_file()), None)
    if shared is None:
        raise ImportError(
            "Install BExplains RefPacks (registry ID: bexplains-refpacks) alongside "
            "MiniMax H3 RefPacks, then restart ComfyUI."
        )
    spec = importlib.util.spec_from_file_location("bexplains_refpacks", shared, submodule_search_locations=[str(shared.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./web"
__version__ = "1.0.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
