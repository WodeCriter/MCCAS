from pydantic import BaseModel, Field, conint, validator, field_validator
from typing import List, Optional, Literal
from backend.schemas.character import Character
from backend.schemas.presentation_style import PresentationStyle

class BuildRequest(BaseModel):
    """User input for building a script."""
    m_channel_name:str
    m_idea: str
    m_description: str
    m_niche: str
    m_tone: str
    m_platform: str
    m_desired_num_of_sections: int
    m_web_search: bool
    m_characters: List[Character]
    m_desired_length_s: conint(gt=10) = Field(..., description="Desired total video length in seconds.")
    m_language: str = Field("English", description="Output language.")
    m_preferred_styles: Optional[List[PresentationStyle]] = Field(None, description="Allowed styles to pick from.")
    m_audience: str = Field("general", description="Target audience description.")

    @field_validator("actor_names")
    def names_match_count(cls, v, values):
        num = values.get("num_actors")
        if v is not None and len(v) != num:
            raise ValueError("actor_names length must equal num_actors")
        return v