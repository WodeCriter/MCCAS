from __future__ import annotations
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, conint
from backend.script.schemas.build_script_request import BuildRequest
from backend.script.schemas.script_metadata import ScriptMetaData
from backend.script.schemas.script_section_info import ScriptSectionInfo
from backend.script.schemas.script_section import ScriptSection, SectionPlan, PlanResponse
from backend.script.schemas.script import Script
from backend.schemas.character import Character
from backend.schemas.presentation_style import PresentationStyle
from backend.script.schemas.voice_line import VoiceLine

LOW_TEMPERATURE = 0.2
MAX_SHORTS_LENGTH_S= 60
SHORTS_SECTION_RANGE = (1, 3)
REGULAR_SECTION_RANGE = (5, 9)
SHORTS_PREFERRED_FLOW = "(Hook -> Value -> CTA)"
LONG_FORM_PREFERRED_FLOW = "(Intro -> Points -> Outro)"

class StylesResponse(BaseModel):
    styles: List[PresentationStyle]

class LineDraft(BaseModel):
    character_name: str
    text: str

class VoiceDraftResponse(BaseModel):
    # final, ordered voice lines for the section
    voice_lines: List[LineDraft]

# ---------- Config ----------
@dataclass
class ScriptBuilderConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.6
    max_output_tokens: int = 8000
    use_prompt_caching: bool = True


# ---------- Builder ----------
class ScriptBuilder:
    def __init__(self, i_config: ScriptBuilderConfig | None = None):
        load_dotenv()
        self.m_client = OpenAI()
        self.m_config = i_config or ScriptBuilderConfig()

    # ===== Public API =====
    def build(self, i_request: BuildRequest) -> Script:
        """
        1) create metadata
        2) create sections (ScriptSectionInfo list)
        3) generate script per section
        4) return complete Script
        """
        script_metadata = self._build_script_metadata(i_request)        # (1)
        section_info_list = self._plan_sections(i_request)              # (2)
        sections_list = self._compose_sections(i_request, section_info_list) # (3)

        script = Script(m_data=script_metadata, m_sections=sections_list, m_characters=i_request.m_characters)  # (4)
        return script

    # Step 1: Create script metadata
    def _build_script_metadata(self, i_req: BuildRequest) -> ScriptMetaData:
        metadata = ScriptMetaData(
            m_channel_name=i_req.m_channel_name,
            m_title=i_req.m_idea,
            m_description=i_req.m_description,
            m_niche=i_req.m_niche,
            m_tone=i_req.m_tone,
            m_desired_num_of_sections=i_req.m_desired_num_of_sections,
            m_web_search=i_req.m_web_search,
            m_platform=i_req.m_platform,
            m_target_length_s=i_req.m_desired_length_s,
            m_primary_audience=i_req.m_audience,
            m_characters=i_req.m_characters,
        )
        return metadata

    # Step 2: AI chooses number + content + style + web_search
    def _plan_sections(self, i_req: BuildRequest) -> List[ScriptSectionInfo]:
        openai_client: OpenAI = self.m_client
        total = int(i_req.m_desired_length_s)
        is_shorts = (i_req.m_platform.lower() in {"shorts", "tiktok", "reels"}) or total <= MAX_SHORTS_LENGTH_S

        # Let the AI choose; give it smart hints only.
        desired_range = SHORTS_SECTION_RANGE if is_shorts else REGULAR_SECTION_RANGE
        if getattr(i_req, "m_desired_num_of_sections", 0):
            desired_range = (i_req.m_desired_num_of_sections, i_req.m_desired_num_of_sections)

        # PresentationStyle guardrail: present allowed values if provided
        style_options = [str(s) for s in (i_req.m_preferred_styles or [])]

        system = (
            "You are an expert YouTube script outliner.\n"
            "Decide how many sections to use and plan each one thoughtfully.\n"
            f"For shorts , prefer {SHORTS_PREFERRED_FLOW}. "
            f"For long-form, {LONG_FORM_PREFERRED_FLOW}.\n"
            "Pick a presentation_style for each section (choose from the allowed list if provided). "
            "If a section relies on current events, statistics, dates, or prices, mark web_search=true.\n"
            "Return strict JSON that matches the response schema."
        )

        user = (
            f"Idea: {i_req.m_idea}\n"
            f"Niche: {i_req.m_niche}\n"
            f"Audience: {i_req.m_audience}\n"
            f"Tone: {i_req.m_tone}\n"
            f"Language: {i_req.m_language}\n"
            f"Platform: {i_req.m_platform}\n"
            f"Target total length (s): {total}\n"
            f"Desired section count range: {desired_range[0]}..{desired_range[1]}\n"
            f"Allowed presentation styles (optional): {style_options}\n"
            "For each section return: index, length_s, title, talking_points (1–5), presentation_style, web_search (bool)."
        )

        headers = {"x-use-prompt-cache": "true"} if getattr(self.m_config, "use_prompt_caching", True) else None
        resp = openai_client.responses.parse(
            model=self.m_config.model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            temperature=self.m_config.temperature,
            max_output_tokens=self.m_config.max_output_tokens,
            text_format=PlanResponse,
            extra_headers=headers,
        )

        plan: PlanResponse = resp.output_parsed
        self._validate_section_lengths(plan.sections, total)
        enum_styles = self._refine_styles_with_ai(i_req, plan.sections)

        # Convert to your schema: return a list of ScriptSectionInfo (metadata only)
        infos: List[ScriptSectionInfo] = []
        for section_plan, style in zip(plan.sections, enum_styles):
            info = ScriptSectionInfo(
                m_web_search=bool(section_plan.web_search),
                m_index=section_plan.index,
                m_length_s=section_plan.length_s,
                m_character_participants=i_req.m_characters,
                m_title=section_plan.title,
                m_talking_points=section_plan.talking_points,
                m_presentation_style=style,
            )
            infos.append(info)

        return infos

    # Step 3: Create script text
    def _compose_sections(self, i_request: BuildRequest, i_section_infos: List[ScriptSectionInfo]) -> List[ScriptSection]:
        """Create final ScriptSection objects: research (optional) ➜ voice lines ➜ script text."""
        shorts = (i_request.m_platform.lower() in {"shorts", "tiktok", "reels"} or int(i_request.m_desired_length_s) <= 60)
        sections: List[ScriptSection] = []

        # Precompute speaker names and mapping
        speaker_names = [self._char_name(c) for c in i_request.m_characters]
        by_name = {self._char_name(c): c for c in i_request.m_characters}

        for info in i_section_infos:

            budget_words = self._approximate_word_budget(info.m_length_s, shorts)

            system = (
                "You are a senior scriptwriter and dialogue editor. "
                "Write natural, speakable voice lines that fit the target word budget. "
                "Lines should be concise and flow logically."
            )
            # Clear, strict instructions for speaker labels when multi-actor
            voice_rules = (
                f"When assigning lines, 'voice_actor' MUST be one of: {speaker_names}."
            )

            user = (
                    f"Language: {i_request.m_language}\n"
                    f"Niche: {i_request.m_niche}\n"
                    f"Tone: {i_request.m_tone}\n"
                    f"Audience: {i_request.m_audience}\n"
                    f"Platform: {i_request.m_platform}\n"
                    f"Section index: {info.m_index}\n"
                    f"Section title: {info.m_title}\n"
                    f"Presentation style: {info.m_presentation_style}\n"
                    f"Talking points: {info.m_talking_points}\n"
                    f"{voice_rules}\n"
                    f"Target budget: ~{budget_words} words total;"
                    "Write directly the lines that will be spoken; avoid meta-instructions or stage directions."
            )
            # Enable web search only when requested on this section
            tool_kwargs = {}
            if getattr(info, "m_web_search", False):
                tool_kwargs["tools"] = [{"type": "web_search"}]  # enable the tool
                tool_kwargs["tool_choice"] = "auto"

            headers = {"x-use-prompt-cache": "true"} if self.m_config.use_prompt_caching else None
            draft = self.m_client.responses.parse(
                model=self.m_config.model,
                input=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
                temperature=self.m_config.temperature,
                max_output_tokens=self.m_config.max_output_tokens,
                text_format=VoiceDraftResponse,
                extra_headers=headers,
                **tool_kwargs,
            ).output_parsed

            # --- (c) map line drafts to VoiceLine objects (assign actor for single-speaker)
            final_lines: List[VoiceLine] = []
            if draft.voice_lines:
                for ln in draft.voice_lines:
                    # choose actor

                    character_name = by_name.get((ln.character_name or "").strip()) or i_request.m_characters[0].m_name
                    final_lines.append(VoiceLine(m_character=character_name, m_text=ln.text))
            else:
                # fallback: single blob
                character_name = i_request.m_characters[0].m_name
                text = (draft.section_text or "").strip()
                final_lines.append(VoiceLine(m_character=character_name, m_text=text or ""))

            # --- (d) assemble the raw script text (what the audience hears)
            # You asked for "the script itself is the raw text being said by the characters"
            # so we concatenate the spoken line texts (no speaker tags) in order.
            script_text = " ".join([vl.m_text for vl in final_lines]).strip()

            # --- (e) build the ScriptSection
            section = ScriptSection(m_metadata=info, m_script_text=script_text, voice_lines=final_lines)
            sections.append(section)

        return sections

    def _refine_styles_with_ai(self, i_req: BuildRequest, i_plans: List[SectionPlan]) -> List[PresentationStyle]:
        """
        Take coarse plans and ask the model to choose EXACT enum values for each section.
        Returns: List[PresentationStyle]
        """
        # Prepare a compact brief for the model
        brief = [
            {
                "index": p.index,
                "length_s": p.length_s,
                "title": p.title,
                "talking_points": p.talking_points,
                # 'presentation_style' here is only a hint; we'll re-pick from enums
                "style_hint": p.presentation_style,
            }
            for p in i_plans
        ]

        allowed = [ps.value for ps in PresentationStyle]

        system = (
            "You are a YouTube editorial lead. For each section, choose exactly ONE presentation style "
            "from the allowed list. Match the style to the title, length, and talking points.\n"
            "Return strict JSON with a 'styles' array aligned to the sections order."
        )
        user = (
            f"Allowed styles: {allowed}\n"
            f"Language: {i_req.m_language}\n"
            f"Niche: {i_req.m_niche}\n"
            f"Tone: {i_req.m_tone}\n"
            f"Audience: {i_req.m_audience}\n"
            f"Platform: {i_req.m_platform}\n"
            f"Sections: {brief}"
        )

        headers = {"x-use-prompt-cache": "true"} if self.m_config.use_prompt_caching else None
        resp = self.m_client.responses.parse(
            model=self.m_config.model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            temperature=LOW_TEMPERATURE,
            max_output_tokens=self.m_config.max_output_tokens,
            text_format=StylesResponse,
            extra_headers=headers,
        )

        styles: List[PresentationStyle] = resp.output_parsed.styles

        # Safety: align length to plans; default to 'explanatory' on mismatch
        if len(styles) != len(i_plans):
            styles = (styles + [PresentationStyle.EXPLANATORY] * len(i_plans))[: len(i_plans)]
        return styles

    def _validate_section_lengths(self, i_section_plans: List[SectionPlan], total: int):
        """Ensure lengths sum to `total` and each >= 1s."""
        for i, p in enumerate(i_section_plans, start=1):
            p.index = i
            try:
                p.length_s = max(1, int(p.length_s))
            except Exception:
                p.length_s = 1

        return

    def _char_name(self, c):
        return getattr(c, "m_name", getattr(c, "name", "")) or "Speaker"

    def _approximate_word_budget(self, length_s: int, shorts: bool) -> int:
        # soft target; you said timing will drift later anyway
        wps = 2.0 if shorts else 2.6
        return max(20, int(length_s * wps))