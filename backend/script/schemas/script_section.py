from pydantic import BaseModel, Field, conint
from typing import List
from backend.script.schemas.script_section_info import ScriptSectionInfo
from backend.script.schemas.voice_line import VoiceLine

class ScriptSection(BaseModel):
    """A section of the video with structure and generated content."""
    # Data
    m_metadata: ScriptSectionInfo
    # Generated content
    m_script_text: str = Field(..., description="Narration/content for this section, plain text.")
    voice_lines: List[VoiceLine] = Field(default_factory=list, description="Lines split by voice actor.")

class SectionPlan(BaseModel):
    index: conint(ge=1)
    length_s: conint(ge=1)
    title: str
    talking_points: List[str]
    presentation_style: str
    web_search: bool

class PlanResponse(BaseModel):
    sections: List[SectionPlan]