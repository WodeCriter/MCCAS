from enum import Enum

class PresentationStyle(str, Enum):
    """How a section should be presented."""
    NARRATIVE = "narrative"         # Story-driven delivery
    EXPLANATORY = "explanatory"     # Educational or tutorial
    LISTICLE = "listicle"           # Numbered list / countdown format
    DEBATE = "debate"               # Two voices arguing points
    INTERVIEW = "interview"         # Host + guest Q&A style
    NEWS = "news"                   # Anchor-style delivery