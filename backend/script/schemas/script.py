from pydantic import BaseModel, Field, conint
from typing import List
from backend.script.schemas.script_section import ScriptSection
from .script_metadata import ScriptMetaData
from backend.schemas.character import Character

class Script(BaseModel):
    """The full video script outline and content."""
    m_data: ScriptMetaData
    m_sections: List[ScriptSection] = Field(default_factory=list)
    m_characters: List[Character] = Field(default_factory=list)