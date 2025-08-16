from pydantic import BaseModel, Field


class VoiceLine(BaseModel):
    """
    A single spoken line assigned to a voice actor.
    """
    character: str = Field(
        ...,
        min_length=1,
        description="Name/identifier of the character.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Exact line to be spoken."
    )