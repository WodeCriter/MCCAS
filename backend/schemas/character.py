from pydantic import BaseModel, Field
class Character(BaseModel):
    """Represents a Character/actor in the script."""
    m_name: str = Field(..., description="Name or identifier of the voice actor.")