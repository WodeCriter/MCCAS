from pydantic import BaseModel, Field, conint
from typing import List
from backend.script.schemas.script_section import ScriptSection
from backend.schemas.character import Character

class ScriptMetaData(BaseModel):
    """The full video script outline and content."""
    m_channel_name: str
    m_title: str
    m_description: str
    m_niche: str
    m_tone: str
    m_platform: str
    m_desired_num_of_sections: int
    m_web_search: bool
    m_target_length_s: conint(gt=0)
    m_intro:str = None
    m_primary_audience: str = Field("general", description="Who the video is for.")
    m_calls_to_action: str = Field("Like & subscribe for more!", description="CTA line for the ending.")
    m_characters: List[Character] = Field(default_factory=list)