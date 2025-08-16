from pydantic import BaseModel, Field
from typing import List
from backend.script.schemas.script_section import ScriptSection
from .script_metadata import ScriptMetaData
from backend.schemas.character import Character
from collections import OrderedDict

class Script(BaseModel):
    """The full video script outline and content."""
    m_data: ScriptMetaData
    m_sections: List[ScriptSection] = Field(default_factory=list)
    m_characters: List[Character] = Field(default_factory=list)



    def export_plain_text(self) -> str:
        """
        1) Plain script: only the spoken lines, in chronological order.
        Returns a single string suitable for writing to a .txt file.
        """
        out_lines = []
        for section in self.m_sections:
            for vl in section.voice_lines:
                text = (getattr(vl, "m_text", getattr(vl, "text", "")) or "").strip()
                if text:
                    out_lines.append(text)
        return "\n".join(out_lines)

    def export_with_speakers(self) -> str:
        """
        2) Script with [character name] above each line.
        Each block is two lines: the speaker tag on its own line, then the line text.
        Blank line separates blocks for readability.
        """
        blocks = []
        for section in self.m_sections:
            for vl in section.voice_lines:
                name = (getattr(vl, "m_character", getattr(vl, "voice_actor", "")) or "").strip()
                text = (getattr(vl, "m_text", getattr(vl, "text", "")) or "").strip()
                if not text:
                    continue
                if name:
                    blocks.append(f"[{name}]\n{text}")
                else:
                    # Fallback: no name present
                    blocks.append(text)
        return "\n\n".join(blocks)

    def export_grouped_by_character(self) -> str:
        """
        3) Lines grouped by character, but preserving the *original order* via indices.
           - Groups are ordered by first appearance in the script.
           - Inside each group, lines appear in chronological order.
           - We include both a global line number and a (section,line) pair so you can reconstruct.
        Output example:
            [Host]
            0001 (S1:L1) Hello!
            0004 (S2:L2) Another line...
        """
        groups = OrderedDict()  # character_name -> list[(global_idx, sec_idx, line_idx, text)]
        global_idx = 1

        for sec_idx, section in enumerate(self.m_sections, start=1):
            for line_idx, vl in enumerate(section.voice_lines, start=1):
                name = (getattr(vl, "m_character", getattr(vl, "voice_actor", "")) or "").strip() or "Unknown"
                text = (getattr(vl, "m_text", getattr(vl, "text", "")) or "").strip()
                if not text:
                    continue
                if name not in groups:
                    groups[name] = []
                groups[name].append((global_idx, sec_idx, line_idx, text))
                global_idx += 1

        # Render
        chunks = []
        for name, lines in groups.items():
            chunks.append(f"[{name}]")
            for gidx, sidx, lidx, text in lines:
                chunks.append(f"{gidx:04d} (S{sidx}:L{lidx}) {text}")
            chunks.append("")  # blank line between groups
        return "\n".join(chunks).rstrip()