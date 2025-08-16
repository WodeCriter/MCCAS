from pydantic import BaseModel, Field, conint
from typing import List
from backend.schemas.character import Character
from backend.schemas.presentation_style import PresentationStyle
class ScriptSectionInfo(BaseModel):
    """A section of the video with structure and generated content."""
    m_web_search:bool
    m_index: conint(ge=1) = Field(..., description="1-based position in the outline.")
    m_length_s: conint(ge=0) = Field(..., description="Length time of this section within the video, in seconds.")
    m_character_participants: List[Character] = Field(default_factory=list)
    m_title: str = Field(..., description="Short heading for this section.")
    m_talking_points: List[str] = Field(default_factory=list, description="Lines split by voice actor.")
    m_presentation_style: PresentationStyle = Field(..., description="How this section should be presented.")