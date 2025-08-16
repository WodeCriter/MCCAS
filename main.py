import json
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
    Returns List[Character].
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[Character] = []
        for item in raw:
            if isinstance(item, str):
                out.append(Character(m_name=item))
            elif isinstance(item, dict):
                out.append(Character(**item))
            else:
                raise ValueError(f"Unsupported character entry: {item!r}")
        return out
    raise ValueError("m_characters must be a list (of names or dicts).")


def _maybe_map_user_friendly_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    If YAML uses human-friendly keys, map them to BuildRequest fields.
    No-op if it already uses m_* keys.
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
    """Serialize pydantic v2/v1 or dataclasses to a plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)  # Pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=True)        # Pydantic v1
    try:
        from dataclasses import asdict
        return asdict(obj)
    except Exception:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


# ---------- main (no argv) ----------
def main() -> None:
    load_dotenv()

    yaml_path = Path("brands/pienantial_trends/youtube/test_script.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    # Optional: builder settings inside YAML
    builder_cfg_block = raw.pop("builder_config", {}) or {}
    builder_cfg = ScriptBuilderConfig(
        model=builder_cfg_block.get("model", "gpt-4.1-mini"),
        temperature=float(builder_cfg_block.get("temperature", 0.6)),
        max_output_tokens=int(builder_cfg_block.get("max_output_tokens", 8000)),
        use_prompt_caching=bool(builder_cfg_block.get("use_prompt_caching", True)),
    )

    # Map human-friendly keys -> m_* (if needed)
    data = _maybe_map_user_friendly_keys(raw)

    # Coerce characters
    if "m_characters" in data:
        data["m_characters"] = _coerce_characters(data["m_characters"])

    # Build request & run
    request = BuildRequest(**data)
    builder = ScriptBuilder(builder_cfg)
    script = builder.build(request)

    # Write JSON next to the YAML (e.g., Brands/.../test_shcema.json)
    out_path = yaml_path.with_suffix(".json")
    out_path.write_text(json.dumps(_pydantic_dump(script), ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional: quick preview if builder provides it
    if hasattr(builder, "quick_preview"):
        print("\n--- Preview ---")
        print(builder.quick_preview(script))


if __name__ == "__main__":
    main()