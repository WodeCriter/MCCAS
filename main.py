from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from backend.script.script_builder import ScriptBuilder, ScriptBuilderConfig
from backend.script.schemas.build_script_request import BuildRequest
from backend.schemas.character import Character


# ---------- helpers ----------
def _coerce_characters(raw: Any) -> List[Character]:
    """
    Accepts either a list of strings (names) or a list of dicts compatible with Character.
    Returns a list[Character].
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[Character] = []
        for item in raw:
            if isinstance(item, str):
                out.append(Character(name=item))
            elif isinstance(item, dict):
                out.append(Character(**item))
            else:
                raise ValueError(f"Unsupported character entry: {item!r}")
        return out
    raise ValueError("m_characters must be a list (of names or dicts).")


def _maybe_map_user_friendly_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    If your YAML uses human-friendly keys, map them to your BuildRequest fields.
    If you already use m_* keys, this is a no-op.
    """
    keymap = {
        "idea": "m_idea",
        "desired_length_s": "m_desired_length_s",
        "desired_num_of_sections": "m_desired_num_of_sections",
        "channel_name": "m_channel_name",
        "description": "m_description",
        "niche": "m_niche",
        "tone": "m_tone",
        "platform": "m_platform",
        "audience": "m_audience",
        "language": "m_language",
        "web_search": "m_web_search",
        "preferred_styles": "m_preferred_styles",
        "characters": "m_characters",
    }
    mapped = {}
    for k, v in data.items():
        mapped[keymap.get(k, k)] = v
    return mapped


def _pydantic_dump(obj: Any) -> Dict[str, Any]:
    """
    Robust serializer for Pydantic v1/v2 or dataclasses.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)  # Pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=True)  # Pydantic v1
    try:
        from dataclasses import asdict  # type: ignore
        return asdict(obj)  # dataclass fallback
    except Exception:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


# ---------- main ----------
def main(argv: List[str] | None = None) -> None:
    """
    Usage:
      python main_build_from_yaml.py --config config.yaml --out script.json [--preview]
    """
    parser = argparse.ArgumentParser(description="Build a YouTube/Shorts script from a YAML request.")
    parser.add_argument("--config", required=True, help="Path to YAML config with BuildRequest fields.")
    parser.add_argument("--out", default="script.json", help="Path to write the resulting Script JSON.")
    parser.add_argument("--preview", action="store_true", help="Print a short preview to stdout.")
    args = parser.parse_args(argv)

    load_dotenv()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"YAML config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Optional builder config block in YAML
    builder_cfg_block = raw.pop("builder_config", {}) or {}
    builder_cfg = ScriptBuilderConfig(
        model=builder_cfg_block.get("model", "gpt-4.1-mini"),
        temperature=float(builder_cfg_block.get("temperature", 0.6)),
        max_output_tokens=int(builder_cfg_block.get("max_output_tokens", 8000)),
        use_prompt_caching=bool(builder_cfg_block.get("use_prompt_caching", True)),
    )

    # Map user-friendly keys to m_* keys if needed
    data = _maybe_map_user_friendly_keys(raw)

    # Coerce characters (names or dicts) -> List[Character]
    if "m_characters" in data:
        data["m_characters"] = _coerce_characters(data["m_characters"])

    # Build the request (Pydantic validation enforces required fields)
    request = BuildRequest(**data)

    # Build the script
    builder = ScriptBuilder(builder_cfg)
    script = builder.build(request)

    # Serialize and write to JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_pydantic_dump(script), f, ensure_ascii=False, indent=2)

    if args.preview:
        # Try a quick preview if available on your builder (optional)
        preview = getattr(builder, "quick_preview", None)
        if callable(preview):
            print("\n--- Preview ---")
            print(preview(script))
        else:
            print("\nPreview not available on this builder.")


if __name__ == "__main__":
    main()