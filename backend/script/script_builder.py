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

class StylesResponse(BaseModel):
    styles: List[PresentationStyle]

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
        3) generate script per section (+ character annotations when >1)
        4) build VoiceLine objects from annotations (or allocate to sole character)
        5) return complete Script
        """
        script_metadata = self._build_script_metadata(i_request)# (1)
        sections = self._plan_sections(i_request)               # (2)
        sections = self._draft_sections(i_request, sections)    # (3)
        sections = self._make_voice_lines(i_request, sections)  # (4)

        script = Script(m_data=script_metadata, m_sections=sections, m_characters=i_request.m_characters)  # (5)
        return script

    # Create script metadata
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

    # ===== Step 2: AI chooses number + content + style + web_search; returns LIST[ScriptSectionInfo] =====
    def _plan_sections(self, i_req: BuildRequest) -> List[ScriptSectionInfo]:
        openai_client: OpenAI = self.m_client
        total = int(i_req.m_desired_length_s)
        is_shorts = (i_req.m_platform.lower() in {"shorts", "tiktok", "reels"}) or total <= 60

        # Let the AI choose; give it smart hints only.
        desired_range = (1, 3) if is_shorts else (5, 9)
        if getattr(i_req, "m_desired_num_of_sections", 0):
            desired_range = (i_req.m_desired_num_of_sections, i_req.m_desired_num_of_sections)

        # PresentationStyle guardrail: present allowed values if provided
        style_options = [str(s) for s in (i_req.m_preferred_styles or [])]

        system = (
            "You are an expert YouTube script outliner.\n"
            "Decide how many sections to use and plan each one thoughtfully.\n"
            "For shorts (<=60s), prefer 1–3 sections (Hook -> Value -> CTA). "
            "For long-form, prefer 5–9 sections (Intro, Points, Outro).\n"
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
            "For each section return: index, length_s, title, talking_points (2–5), presentation_style, web_search (bool)."
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
        fixed = self._normalize_lengths(plan.sections, total)
        enum_styles = self._refine_styles_with_ai(i_req, fixed)

        # Convert to your schema: return a list of ScriptSectionInfo (metadata only)
        infos: List[ScriptSectionInfo] = []
        for section_plan, style in zip(fixed, enum_styles):
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

    # ===== Step 3: draft script text (+ annotations when >1 character) =====
    def _draft_sections(self, req: BuildRequest, sections: List[ScriptSection]) -> List[ScriptSection]:
        # Build a compact brief for the model
        brief = [
            dict(
                m_index=s.m_metadata.m_index,
                t0=s.m_metadata.m_length_s,
                t1=s.m_metadata.m_time_end_s,
                title=s.m_metadata.m_title,
                style=s.m_metadata.m_presentation_style,
            )
            for s in sections
        ]

        multi_speaker = len(req.m_characters) > 1
        speaker_list = [c.name for c in req.m_characters]

        system = (
            "You are an elite YouTube scriptwriter. Write tight, engaging narration per section.\n"
            "Keep the pace natural; avoid filler.\n"
            "If multiple characters are provided, you MUST label each line with 'CharacterName: text'.\n"
            "Return JSON matching the response schema."
        )
        user = (
            f"Language: {req.m_language}\n"
            f"Niche: {req.m_niche}\n"
            f"Tone: {req.m_tone}\n"
            f"Audience: {req.m_audience}\n"
            f"Platform: {req.m_platform}  (<=60s means shorts pacing)\n"
            f"Characters: {speaker_list}\n"
            f"Sections (with time windows): {brief}\n\n"
            "For each section:\n"
            "- Produce m_script_text (a single block of narration fitting its time window)\n"
            f"- {'Also include lines[] with speaker-labeled entries' if multi_speaker else 'lines[] may be omitted'}"
        )

        headers = {"x-use-prompt-cache": "true"} if self.m_config.use_prompt_caching else None
        resp = self.m_client.responses.parse(
            model=self.m_config.model,
            input=[{"role": "system", "content": system},
                   {"role": "user", "content": user}],
            temperature=self.m_config.temperature,
            max_output_tokens=self.m_config.max_output_tokens,
            response_format=DraftResponse,
            extra_headers=headers,
        )
        draft = resp.output_parsed
        by_idx: Dict[int, SectionDraft] = {d.m_index: d for d in draft.sections}

        for s in sections:
            d = by_idx.get(s.m_metadata.m_index)
            if d:
                s.m_script_text = d.m_script_text
                # Temporarily stash parsed 'lines' in a private attr for step 4
                setattr(s, "_draft_lines", d.lines or [])

        return sections

    # ===== Step 4: build VoiceLine objects =====
    def _make_voice_lines(self, req: BuildRequest, sections: List[ScriptSection]) -> List[ScriptSection]:
        # map name -> Character
        by_name = {c.name: c for c in req.m_characters}
        solo: Optional[Character] = req.m_characters[0] if len(req.m_characters) == 1 else None

        for s in sections:
            lines_data: List[Line] = getattr(s, "_draft_lines", []) or []
            final_lines: List[VoiceLine] = []

            if lines_data:
                for ln in lines_data:
                    actor = by_name.get((ln.voice_actor or "").strip())
                    if not actor and solo:
                        actor = solo
                    if not actor:
                        # Fallback: leave it unassigned or skip; here we fallback to first character.
                        actor = req.m_characters[0]
                    final_lines.append(VoiceLine(voice_actor=actor, text=ln.text))
            else:
                # No annotations returned
                if solo:
                    final_lines.append(VoiceLine(voice_actor=solo, text=s.m_script_text))
                else:
                    # Multi-speaker but no annotations: round-robin sentences
                    sentences = [seg.strip() for seg in s.m_script_text.split(".") if seg.strip()]
                    if not sentences:
                        sentences = [s.m_script_text]
                    i = 0
                    for sent in sentences:
                        final_lines.append(VoiceLine(voice_actor=req.m_characters[i % len(req.m_characters)],
                                                     text=(sent + "." if not sent.endswith(".") else sent)))
                        i += 1

            s.voice_lines = final_lines
            if hasattr(s, "_draft_lines"):
                delattr(s, "_draft_lines")

        return sections

    def _refine_styles_with_ai(self, i_req: BuildRequest, plans: List[SectionPlan]) -> List[PresentationStyle]:
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
            for p in plans
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
        if len(styles) != len(plans):
            styles = (styles + [PresentationStyle.EXPLANATORY] * len(plans))[: len(plans)]
        return styles

    def _normalize_lengths(self, plans: List[SectionPlan], total: int) -> List[SectionPlan]:
        """Ensure lengths sum to `total` and each >= 1s. Extend/trim the last section if needed."""
        if not plans:
            return plans
        s = sum(p.length_s for p in plans)
        if s == 0:
            # make one full-length section if something is off
            return [
                SectionPlan(index=1, length_s=total, title="Main", talking_points=[], presentation_style="explanatory",
                            web_search=False)]
        diff = total - s
        # adjust the last section by the diff
        plans[-1].length_s = max(1, plans[-1].length_s + diff)
        # if we overshot negative, trim earlier ones as needed (rare)
        while sum(p.length_s for p in plans) > total and len(plans) > 1:
            overflow = sum(p.length_s for p in plans) - total
            take = min(overflow, plans[-1].length_s - 1)
            plans[-1].length_s -= take
            if take < overflow and len(plans) > 2:
                plans[-2].length_s = max(1, plans[-2].length_s - (overflow - take))
        # reindex 1..N (in case the model didn't)
        for i, p in enumerate(plans, start=1):
            p.index = i
        return plans