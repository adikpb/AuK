from __future__ import annotations

import argparse
import math
import os

import gradio as gr
import torch

from auk.infer.infer_auk import AukInfer, get_gen_duration
from auk.infer.pe import PromptEnhancer, PromptEnhancerError


# Static examples derived from the public AuK demo sample list.
# Each row is: (audio path relative to the repository root, instruction, target duration seconds).
DEMO_EXAMPLE_GROUPS: dict[str, dict[str, list[tuple[str | None, str, float]]]] = {
    "Speech Generation": {
        "Zero-shot TTS": [
            (
                "assets/demo-input-audio/zero-shot-tts/ref.wav",
                "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'",
                6.0,
            ),
        ],
        "Instruct TTS": [
            (
                None,
                "Say the following in the voice described here: “一位雄才大略、性格复杂的乱世枭雄，以略显沙哑却极有穿透力的中年男声说话。语气自信、果断，带着审视人心的敏锐感。讲话时节奏变化明显，可以先压低声音缓缓铺垫，再突然加重关键字。既有豪迈，也隐约带着危险与猜疑”, and say: “宁可我负天下人，休教天下人负我。”",
                3.36,
            ),
            (
                None,
                "Say the following in the voice described here: “一位中年女性，在面对面质问多年前抛弃自己的人时，用沙哑、干涩的嗓音倾诉三十年压抑的恨意与委屈，仿佛眼泪已经流尽。语速缓慢而沉稳，字字清晰冰冷，整体音量中等偏低，但说到愤恨处声音微微发颤、压抑着哽咽，音色苍凉而有力，句尾带着渗骨的决绝，最后一句陡然拔高、几乎用尽全身力气砸出来”, and say: “哼，我的眼泪早哭干了，我没有委屈，我有的是恨，是悔，是三十年一天一天我自己受的苦。你大概已经忘了你做的事了！”",
                15.0,
            ),
            (
                None,
                "Say the following in the voice described here: “一位莎士比亚戏剧风格的经典反派，正在揭露真相前细细品味这一刻。声音是浑厚且富有戏剧张力的男中音，胸腔共鸣饱满，语速从容舒缓，带着刻意的停顿与强调；辅音被夸张地滚动、咬字清晰而富有舞台感。语调先佯装甜蜜温柔，几乎带着宠溺，随后在词句间骤然转冷，锋利如刀刃，每个字都缠绕着危险而愉悦的得意之情”, and say: “You see, my dear, the trap was never meant for you. It was always meant for him.”",
                7.0,
            ),
            (
                None,
                "Say the following in the voice described here: “仿佛在葬礼上宣布噩耗一般，声音轻柔而哽咽，强忍着泪水，话语断断续续、每句之间都有停顿，字字愈发沉重，最后一句的尾音被悲伤撕裂、颤抖着透出压抑不住的哭腔。语调缓慢低沉，气息不稳，清晰度因哽咽而略受影响，整体氛围凝重而哀恸”, and say: “He always said he'd come back. He always kept his word. Until now.”",
                7.0,
            ),
        ],
    },
    "Content Editing": {
        "Speech Content Editing — Replace": [
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
                7.0,
            ),
        ],
        "Content Insert (before / after)": [
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Add 'truly' before 'accepting' in the speech.",
                7.4,
            ),
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Add 'with courage' after 'dreams unmet'.",
                7.8,
            ),
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "在‘我们’后面加上‘一定会’",
                7.2,
            ),
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "在‘接受’前面加上‘学会’",
                7.2,
            ),
        ],
        "Content Remove (delete)": [
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Remove 'but accepting'.",
                6.5,
            ),
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "删掉‘我们不能拥有的’",
                6.0,
            ),
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Remove 'what we cannot have' after 'accepting'.",
                6.2,
            ),
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "删掉‘但是’后面的‘接受’",
                6.0,
            ),
        ],
        "Lyric Editing": [
            (
                "assets/demo-input-audio/vocal-edit/vocaledit-en-1-input.wav",
                "Replace “rear view” with “like you” in the lyrics",
                5.44,
            ),
            (
                "assets/demo-input-audio/vocal-edit/vocaledit-en-1-input.wav",
                "Change \"rear view\" to \"like you\" in the vocal recording.",
                5.44,
            ),
        ],
    },
    "Acoustic Editing": {
        "Pitch Editing": [
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低3个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低2个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低1个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高1个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高2个半音。", 5.12),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高3个半音。", 5.12),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "Lower the pitch by 2 semitones.", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "Raise the pitch by 2 semitones.", 5.0),
        ],
        "Speed Editing": [
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为0.5倍。", 20.6),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为0.75倍。", 13.74),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为1.25倍。", 8.24),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为1.5倍。", 6.86),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为2.0倍。", 5.16),
        ],
        "Volume Editing": [
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低15分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低10分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低5分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高5分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高10分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高15分贝。", 3.78),
        ],
    },
    "Paralinguistic Editing": {
        "Emotion Editing": [
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a happy tone", 6.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a sad tone", 7.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a angry tone", 6.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a afraid tone", 7.22),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Change the emotion to excited.", 6.6),
        ],
        "Timbre Editing": [
            (
                "assets/demo-input-audio/vc/vc-1-input.wav",
                "Keep the words and change the timbre to: “这位说话人的声音低沉而浑厚，语速平稳，吐字清晰。他的说话风格沉稳而富有思考，带有平静的反思特质。”",
                6.56,
            ),
        ],
        "De-accent": [
            ("assets/demo-input-audio/accent/accent-anhui-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 4.0),
            (
                "assets/demo-input-audio/accent/accent-dongbei-input.wav",
                "请去掉这段语音里的方言口音，保持说话人音色一致。",
                17.52,
            ),
            ("assets/demo-input-audio/accent/accent-fujian-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 4.06),
            ("assets/demo-input-audio/accent/accent-hubei-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 2.2),
            ("assets/demo-input-audio/accent/accent-hunan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 3.52),
            ("assets/demo-input-audio/accent/accent-sichuan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 6.74),
            ("assets/demo-input-audio/accent/accent-tibetan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 3.22),
        ],
        "Nonverbal Editing": [
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a breath before “We tested”", 10.44),
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a sneeze before “only one of them”", 12.0),
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a pause after “only one of them”", 11.0),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the humming", 22.0),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the hiss", 22.82),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the sobbing", 21.0),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a sigh before “这个月”", 13.0),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a filler before “主要的缺口”", 12.38),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a cough before “再决定”", 13.0),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the laughter", 12.58),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the gasp of surprise", 12.72),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the sharp inhale", 12.62),
        ],
        "Whisper — To Whisper": [
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "Turn this into a whisper", 8.36),
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "用小声耳语的方式把这段话说出来。", 8.36),
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "Convert this speech into a soft whisper while preserving the speaker and content.", 8.36),
        ],
        "Whisper — To Normal": [
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "Convert this whispered speech into a normal speaking voice while preserving the speaker and content.", 8.36),
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "把这段耳语转换成正常说话的声音。", 8.36),
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "把这段耳语转换成正常说话的声音，保持说话人和内容不变。", 8.36),
        ],
    },
    "Enhancement & Separation": {
        "Speech Enhancement": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Remove the background noise and make the voice cleaner", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "请只去除背景噪声，保留其他内容，输出等长结果。", 5.0),
        ],
        "Speech Separation (by order)": [
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the first speaker", 28.0),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "Keep only the second speaker", 18.24),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "只保留第一个开始说话的人，去掉其余说话人。", 19.28),
            ("assets/demo-input-audio/ss/en-1-input.wav", "Please keep the second speaker to start talking and remove the other speakers.", 28.0),
        ],
        "Target Speaker Extraction (by content)": [
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the speaker who says “get what”", 28.0),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "Keep only the speaker who says “警队规矩”", 19.28),
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the speaker who says \"money\" and remove all other speakers.", 22.0),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "请只保留说“规矩”的人，去掉其他说话人。", 19.28),
        ],
        "Vocal Extraction — Singing Only": [
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "Extract the vocals and remove the accompaniment",
                10.88,
            ),
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "Keep only the singing voice and remove everything else.",
                10.88,
            ),
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "请只保留歌声，其余声音都去掉。",
                10.88,
            ),
        ],
        "Vocal Extraction — All Voices": [
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "Keep all human voices, including speech and singing, and remove everything else.",
                10.88,
            ),
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "请保留所有人声，包括说话和歌唱，其余声音都去掉。",
                10.88,
            ),
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "Extract all human voices (speech + singing) and drop the accompaniment.",
                10.88,
            ),
        ],
        "Audio Quality Enhancement": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Improve the audio quality and make it clearer", 5.0),
        ],
        "Speech Quality Restoration": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Restore the bandwidth and make the audio wideband and clear (super-resolution).", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "请对这段语音做超分辨率/带宽扩展处理，恢复被削掉的高频成分，输出宽带纯净人声。", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Remove the telephone effect and restore natural clear speech.", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "请消除这段音频的电话音色，恢复自然清晰的人声。", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Fix the clipping distortion and restore clean speech.", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Repair the dropout and restore natural, clear speech.", 5.0),
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Improve the audio quality and restore clear speech, removing the muffled effect.", 5.0),
        ],
    },
}

# variant label -> checkpoint path (filled in main() from CLI args)
CKPT_PATHS: dict[str, str] = {}
CONFIG_PATHS: dict[str, str | None] = {}
# variant label -> loaded engine, populated lazily on first use and reused after
ENGINES: dict[str, AukInfer] = {}
ENGINE_KWARGS: dict = {}
DEVICE_PATHS: dict[str, str | None] = {}

BASE_LABEL = "AuK (Base)"
FLASH_LABEL = "AuK-Flash ⚡"

SAMPLING_PRESETS = {
    BASE_LABEL: {"nfe": 32, "cfg": 2.0, "interactive": True},
    FLASH_LABEL: {"nfe": 4, "cfg": 0.0, "interactive": False},
}

LOUDNESS_LIMIT_TASK_TYPES = frozenset({"extract_vocals", "vocal_edit"})
VOCAL_OUTPUT_TARGET_LUFS = -14.0
VOCAL_OUTPUT_PEAK_CEILING = 0.95

# Category metadata for hierarchy / iconography
CATEGORY_META: dict[str, dict[str, str]] = {
    "Speech Generation": {"icon": "✦", "desc": "Synthesize speech from text — clone a voice or describe one."},
    "Content Editing": {"icon": "✎", "desc": "Rewrite what is said: replace · insert · remove · lyric edit."},
    "Acoustic Editing": {"icon": "◐", "desc": "Precise acoustic controls — pitch · speed · volume."},
    "Paralinguistic Editing": {"icon": "◎", "desc": "Style & delivery — emotion · timbre · accent · breath · whisper."},
    "Enhancement & Separation": {"icon": "⬢", "desc": "Clean up and separate — denoise · separate speakers · extract vocals · restore quality."},
}

DEMO_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,800&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --auk-accent: #6d5bff;
  --auk-accent-2: #ff6b9d;
  --auk-ink: #0f0f14;
  --auk-paper: #fafaf8;
  --auk-line: #e8e6e2;
  --auk-muted: #6b6b78;
}
#auk-hero {
  background: radial-gradient(1200px 400px at 10% -20%, rgba(109,91,255,0.16), transparent 60%),
              radial-gradient(900px 360px at 100% 0%, rgba(255,107,157,0.14), transparent 60%),
              linear-gradient(180deg, #ffffff 0%, #fafaf8 100%);
  border: 1px solid var(--auk-line);
  border-radius: 20px;
  padding: 22px 22px 18px 22px;
  margin-bottom: 14px;
}
#auk-hero h1 {
  font-family: 'Fraunces', serif !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  line-height: 0.95 !important;
  font-size: 30px !important;
  margin: 0 !important;
  color: var(--auk-ink) !important;
}
#auk-hero p {
  font-family: 'Instrument Sans', sans-serif !important;
  color: var(--auk-muted) !important;
  margin-top: 8px !important;
  font-size: 13.5px !important;
  line-height: 1.5 !important;
}
.auk-hero-badges {
  display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
}
.auk-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--auk-line); background: white; color: #2b2b33;
}
.auk-badge.accent {
  background: var(--auk-ink); color: white; border-color: var(--auk-ink);
}
#auk-supported {
  border: 1px dashed var(--auk-line);
  border-radius: 16px;
  padding: 12px 14px;
  background: #fff;
  margin-bottom: 14px;
}
#auk-supported .auk-supported-title {
  font-family: 'Instrument Sans', sans-serif;
  font-weight: 700; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--auk-muted); margin-bottom: 8px;
}
.auk-supported-grid {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 14px;
}
@media (max-width: 900px) { .auk-supported-grid { grid-template-columns: 1fr; } }
.auk-supported-item { font-family: 'Instrument Sans', sans-serif; font-size: 12.5px; line-height: 1.4; color: #2b2b33; }
.auk-supported-item b { font-weight: 700; color: var(--auk-ink); }
.auk-duration-note, .auk-sampling-note { margin-top: -6px; color: var(--body-text-color-subdued); font-size: 12px; }
#auk-main-row { align-items: stretch; gap: 16px; }
#auk-input-column, #auk-output-column {
  display: flex; flex-direction: column; height: 100%;
  background: white; border: 1px solid var(--auk-line); border-radius: 16px; padding: 14px;
}
#auk-input-column { box-shadow: 0 1px 0 rgba(0,0,0,0.03); }
#auk-output-column { background: linear-gradient(180deg, #ffffff 0%, #fcfcff 100%); }
#auk-input-bottom, #auk-output-bottom { margin-top: auto; }
#auk-output-bottom { display: flex; flex-direction: column; }
#auk-input-audio { min-height: 210px !important; }
#auk-output-audio { min-height: 200px !important; }
#auk-pe-output { margin-top: 12px; }
.auk-pe-card textarea { min-height: 92px !important; height: 92px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }
#auk-pe-output .auk-pe-header { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom: 6px; }
#auk-pe-output .auk-pe-header h3 { font-family: 'Instrument Sans', sans-serif; font-weight: 700; font-size: 14px; margin: 0; }
.auk-examples-header { display:flex; align-items:baseline; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-top: 18px; }
.auk-examples-header h2 { font-family: 'Fraunces', serif; font-weight: 700; font-size: 22px; letter-spacing: -0.02em; margin: 0; }
.auk-examples-header p { font-family: 'Instrument Sans', sans-serif; color: var(--auk-muted); font-size: 12.5px; margin: 0; }
.auk-category-note { font-family: 'Instrument Sans', sans-serif; color: var(--auk-muted); font-size: 12px; margin: -4px 0 8px 0; }
.auk-examples-table table tbody tr > td:last-child,
.auk-examples-table table tbody tr > td:last-child * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  overflow-wrap: anywhere !important; word-break: break-word !important;
}
.auk-examples-table button { border-radius: 10px !important; }
#auk-generate-btn { background: var(--auk-ink) !important; border-color: var(--auk-ink) !important; font-family: 'Instrument Sans', sans-serif !important; font-weight: 700 !important; letter-spacing: 0.02em !important; border-radius: 12px !important; padding: 12px 16px !important; }
#auk-generate-btn:hover { filter: brightness(1.05); transform: translateY(-1px); transition: all 140ms ease; }
@media (max-width: 960px) {
  #auk-main-row { flex-direction: column !important; }
  #auk-input-column, #auk-output-column { height: auto; }
  #auk-hero h1 { font-size: 26px !important; }
}
@media (max-width: 640px) {
  #auk-hero { padding: 16px; border-radius: 16px; }
  .auk-supported-grid { gap: 6px; }
}
"""


def default_path(*parts: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, *parts)


def update_sampling_controls(variant: str):
    preset = SAMPLING_PRESETS.get(variant, SAMPLING_PRESETS[BASE_LABEL])
    interactive = preset["interactive"]
    return (
        gr.update(value=preset["nfe"], interactive=interactive),
        gr.update(value=preset["cfg"], interactive=interactive),
    )


def get_engine(variant: str) -> AukInfer:
    if variant not in ENGINES:
        ckpt_path = CKPT_PATHS.get(variant)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise gr.Error(f"{variant} checkpoint not found: {ckpt_path}")
        # config.yaml ships next to the checkpoint (release dirs bundle their own)
        ckpt_dir_config = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.yaml")
        config_path = CONFIG_PATHS.get(variant) or ckpt_dir_config
        if not os.path.isfile(config_path):
            raise gr.Error(f"config.yaml not found next to {variant} checkpoint: {config_path}")
        gr.Info(f"Loading {variant} … (first load takes ~15s)")
        ENGINES[variant] = AukInfer(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=DEVICE_PATHS.get(variant) or ENGINE_KWARGS.get("device"),
            **{key: value for key, value in ENGINE_KWARGS.items() if key != "device"},
        )
    return ENGINES[variant]


def _limit_vocal_output_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    output = waveform.to(torch.float64)
    try:
        import pyloudnorm as pyln  # type: ignore[import-not-found]

        measured_lufs = float(pyln.Meter(sample_rate).integrated_loudness(output.numpy()))
        if math.isfinite(measured_lufs) and measured_lufs > VOCAL_OUTPUT_TARGET_LUFS:
            gain = 10.0 ** ((VOCAL_OUTPUT_TARGET_LUFS - measured_lufs) / 20.0)
            output = output * min(1.0, gain)
    except Exception:  # noqa: BLE001,S110
        pass

    if output.numel():
        peak = float(output.abs().max())
        if math.isfinite(peak) and peak > VOCAL_OUTPUT_PEAK_CEILING:
            output = output * (VOCAL_OUTPUT_PEAK_CEILING / peak)
    return output


def _format_output_audio(out_audio: torch.Tensor, sample_rate: int, task_type: str | None = None):
    waveform = out_audio.detach().to(torch.float64).cpu()
    if task_type in LOUDNESS_LIMIT_TASK_TYPES and waveform.ndim == 2:
        if waveform.shape[0] <= 8:
            waveform = waveform.mean(dim=0)
        elif waveform.shape[1] <= 8:
            waveform = waveform.mean(dim=1)
    elif waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.reshape(-1)
    if waveform.ndim != 1:
        raise gr.Error(f"Expected mono output audio, got shape {tuple(waveform.shape)}.")
    if not torch.isfinite(waveform).all():
        raise gr.Error("Generated audio contains NaN or Inf.")
    if task_type in LOUDNESS_LIMIT_TASK_TYPES:
        waveform = _limit_vocal_output_loudness(waveform, sample_rate)
    pcm16 = torch.round(waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()
    return int(sample_rate), pcm16


def run_generate(variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed, task_type=None):
    if not instruction:
        raise gr.Error("Please provide an instruction.")
    if not audio and not gen_seconds and not gen_text:
        raise gr.Error("Instruct TTS without reference audio needs a target duration or target text to set the length.")

    instruction = instruction.strip()
    content = [{"type": "text", "text": instruction}]
    if audio:
        content.append({"type": "audio", "audio": audio})
    messages = [{"role": "user", "content": content}]

    engine = get_engine(variant)
    # nfe/cfg are used as-is for base AuK; AuK-Flash ignores them (locked 4-step / CFG-off recipe).
    gen_seconds = get_gen_duration(
        audio=(audio or None),
        ref_text=(ref_text or None),
        gen_text=(gen_text or None),
        gen_seconds=(gen_seconds or None),
    )
    out_audio, sr = engine.generate(
        messages,
        audio=(audio or None),
        gen_seconds=gen_seconds,
        nfe=int(nfe),
        cfg_strength=float(cfg),
        seed=(int(seed) if seed is not None else None),
    )
    return _format_output_audio(out_audio, sr, task_type=task_type)


def run_generate_with_pe(use_pe, variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed):
    if not use_pe:
        if not gen_seconds:
            raise gr.Error("Duration must be greater than 0 when Prompt Enhancer is disabled.")
        return run_generate(variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed), None

    prepared = None
    try:
        requested_duration = float(gen_seconds or 0)
        prepared = PromptEnhancer().prepare(
            instruction,
            audio,
            target_duration=requested_duration if requested_duration > 0 else None,
        )
        effective_duration = requested_duration if requested_duration > 0 else prepared.gen_seconds
        generated = run_generate(
            variant,
            prepared.audio,
            prepared.instruction,
            effective_duration,
            prepared.ref_text,
            prepared.gen_text,
            nfe,
            cfg,
            seed,
            prepared.task_type,
        )
        asr = getattr(prepared, "asr", None)
        asr_text = asr.text if asr and asr.text else None
        task = str(getattr(prepared, "task_type", ""))
        operation_subtype = getattr(prepared, "operation_subtype", None)
        if operation_subtype:
            task = f"{task} / {operation_subtype}"
        metadata = {
            "task": task,
            "duration": f"{effective_duration:.2f} s",
            "asr_content": asr_text or "",
            "instruction": prepared.instruction,
        }
        return generated, metadata
    except (PromptEnhancerError, ValueError, FileNotFoundError) as exc:
        raise gr.Error(f"Prompt Enhancer failed: {type(exc).__name__}: {exc}") from None
    finally:
        if prepared is not None:
            prepared.cleanup()


def update_pe_outputs(metadata: dict | None):
    if not metadata:
        return "", "", ""
    return (
        f"Task: {metadata.get('task') or ''}\nTarget Duration: {metadata.get('duration') or ''}",
        str(metadata.get("asr_content") or ""),
        str(metadata.get("instruction") or ""),
    )


def load_demo_example(index: int | None, examples: list[tuple[str | None, str, float]]):
    if index is None or not 0 <= int(index) < len(examples):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    audio, instruction, duration = examples[int(index)]
    return audio, instruction, duration, True, 42


def _resolve_demo_examples(examples: list[tuple[str | None, str, float]]) -> list[tuple[str | None, str, float]]:
    resolved = []
    for audio, instruction, duration in examples:
        audio_path = default_path(*audio.split("/")) if audio else None
        if audio_path is None or os.path.isfile(audio_path):
            resolved.append((audio_path, instruction, duration))
    return resolved


def _demo_dataset_samples(task: str, examples: list[tuple[str | None, str, float]]) -> list[list[str | None]]:
    if task == "Instruct TTS":
        return [[instruction] for _, instruction, _ in examples]
    return [[audio, instruction] for audio, instruction, _ in examples]


def build_demo() -> gr.Blocks:
    available_variants = [label for label, path in CKPT_PATHS.items() if path and os.path.isfile(path)]
    if not available_variants:
        raise RuntimeError("No valid model checkpoint was configured.")
    initial_variant = BASE_LABEL if BASE_LABEL in available_variants else available_variants[0]
    initial_sampling = SAMPLING_PRESETS.get(initial_variant, SAMPLING_PRESETS[BASE_LABEL])
    with gr.Blocks(title="AuK — Foundational Speech Generation & Editing") as demo:
        # --- Hero header ---
        with gr.Column(elem_id="auk-hero"):
            gr.Markdown(
                """
# AuK: An Open-Source Foundational Model for Speech Generation and Editing
<p>Test every AuK capability in one place — ChatML messages, natural-language instructions, and PE auto-duration. Pick an example, tweak the instruction, and generate.</p>
"""
            )
            gr.HTML(
                """<div class="auk-hero-badges">
                    <span class="auk-badge accent">Unified ChatML — text + audio</span>
                    <span class="auk-badge">Base · NFE 32 · CFG 2.0</span>
                    <span class="auk-badge">Flash · 4-step · CFG-off</span>
                    <span class="auk-badge">PE: auto duration when 0</span>
                </div>"""
            )
        # --- Supported tasks overview (improved hierarchy) ---
        with gr.Column(elem_id="auk-supported"):
            gr.Markdown('<div class="auk-supported-title">Supported tasks — 22 tabs across 5 groups</div>')
            gr.HTML(
                """
<div class="auk-supported-grid">
  <div class="auk-supported-item"><b>✦ Speech Generation</b> — Zero-shot TTS · Instruct TTS</div>
  <div class="auk-supported-item"><b>✎ Content Editing</b> — Replace · Insert (before/after) · Remove (delete) · Lyric Editing</div>
  <div class="auk-supported-item"><b>◐ Acoustic Editing</b> — Pitch · Speed · Volume</div>
  <div class="auk-supported-item"><b>◎ Paralinguistic</b> — Emotion · Timbre · De-accent · Nonverbal · Whisper ↔ Normal</div>
  <div class="auk-supported-item"><b>⬢ Enhancement & Separation</b> — Denoise · Separation (by order) · Target Speaker (by content) · Vocals (singing / all voices) · Quality Restoration</div>
  <div class="auk-supported-item"><b>Duration priority</b> — explicit Duration &gt; ref+target text estimate &gt; source length</div>
</div>
"""
            )
        gr.Markdown(
            "> **How to test:** Select an example below → it loads into the left panel → edit the instruction if you like → keep **Use Prompt Enhancer** ON for auto task routing, or turn it OFF and set Duration manually → **Generate**.",
            elem_classes="auk-muted",
        )
        with gr.Row(equal_height=True, elem_id="auk-main-row"):
            with gr.Column(elem_id="auk-input-column"):  # left: inputs (audio + text)
                gr.Markdown("#### Input")
                in_audio = gr.Audio(
                    label="Input audio — leave empty for Instruct TTS",
                    type="filepath",
                    elem_id="auk-input-audio",
                    buttons=["download"],
                )
                in_instr = gr.Textbox(
                    label="Instruction — natural language",
                    placeholder="e.g. “Replace ‘but accepting what we cannot have’ with ‘and living well with dreams unmet’.” or “把这段耳语转换成正常说话的声音。”",
                    lines=4,
                    info="PE will classify the task and rewrite the instruction into the canonical template.",
                )
                with gr.Column(elem_id="auk-input-bottom"):
                    use_pe = gr.Checkbox(
                        value=True,
                        label="Use Prompt Enhancer",
                        info="ON: PE determines task, instruction, and duration (0 = auto). OFF: instruction sent verbatim; Duration must be >0.",
                    )
                    # Keep empty internal states for the existing generation callback
                    # after removing the two manual text boxes from the UI.
                    in_ref_text = gr.State(value="")
                    in_gen_text = gr.State(value="")
            with gr.Column(elem_id="auk-output-column"):  # right: inference settings + output
                gr.Markdown("#### Generation")
                in_variant = gr.Radio(
                    choices=available_variants,
                    value=initial_variant,
                    label="Model variant",
                    info="Flash locks sampling to 4-step / CFG 0.",
                )
                in_secs = gr.Slider(
                    0,
                    30,
                    value=0,
                    step=0.5,
                    label="Target duration (seconds)",
                    info="0 → PE estimates; >0 overrides PE.",
                )
                gr.Markdown(
                    "0 = PE auto-estimation (requires PE). Set explicitly when PE is off.",
                    elem_classes="auk-duration-note",
                )
                with gr.Row():
                    in_nfe = gr.Slider(
                        4,
                        64,
                        value=initial_sampling["nfe"],
                        step=1,
                        label="NFE steps",
                        interactive=initial_sampling["interactive"],
                        scale=2,
                    )
                    in_cfg = gr.Slider(
                        0.0,
                        5.0,
                        value=initial_sampling["cfg"],
                        step=0.1,
                        label="CFG strength",
                        interactive=initial_sampling["interactive"],
                        scale=2,
                    )
                    in_seed = gr.Number(label="Seed", value=42, precision=0, scale=1)
                gr.Markdown(
                    "Base: NFE 32 / CFG 2.0 · Flash: NFE 4 / CFG 0 (locked)",
                    elem_classes="auk-sampling-note",
                )
                gen_btn = gr.Button("Generate  →", variant="primary", elem_id="auk-generate-btn")
                with gr.Column(elem_id="auk-output-bottom"):
                    out_audio = gr.Audio(label="Output audio — 24 kHz mono", elem_id="auk-output-audio", buttons=["download"], interactive=False)
                    pe_metadata_state = gr.State(value=None)

        with gr.Column(elem_id="auk-pe-output"):
            with gr.Row(elem_classes="auk-pe-header"):
                gr.Markdown("### Prompt Enhancer output")
                gr.Markdown("<span style='font-family:JetBrains Mono,monospace;font-size:11px;color:#6b6b78;'>read-only · reflects PE classification</span>")
            with gr.Row(equal_height=True):
                pe_summary = gr.Textbox(
                    label="Summary (task / duration)",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=1,
                )
                pe_asr_content = gr.Textbox(
                    label="ASR Content (source transcript)",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=2,
                )
                pe_instruction = gr.Textbox(
                    label="Canonical Instruction (sent to model)",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=2,
                )

        in_variant.change(
            update_sampling_controls,
            inputs=in_variant,
            outputs=[in_nfe, in_cfg],
            queue=False,
            api_visibility="private",
        )
        generate_event = gen_btn.click(
            run_generate_with_pe,
            inputs=[use_pe, in_variant, in_audio, in_instr, in_secs, in_ref_text, in_gen_text, in_nfe, in_cfg, in_seed],
            outputs=[out_audio, pe_metadata_state],
            api_name="run_generate_with_pe",
        )
        generate_event.then(
            update_pe_outputs,
            inputs=pe_metadata_state,
            outputs=[pe_summary, pe_asr_content, pe_instruction],
            queue=False,
            api_visibility="private",
        )

        # --- Examples ---
        with gr.Column():
            with gr.Row(elem_classes="auk-examples-header"):
                gr.Markdown("## Examples")
                gr.Markdown("<p>Click any row to load it — try editing the instruction and sweeping seeds.</p>")
            gr.Markdown(
                "Coverage includes: **Target Speaker Extraction (by content)**, **Vocal Extraction — All Voices**, **Quality Restoration (bandwidth + telephone/clipping/dropout)**, **Content Insert/Remove**, and **Whisper ↔ Normal**.",
                elem_classes="auk-category-note",
            )
        with gr.Tabs():
            for category, tasks in DEMO_EXAMPLE_GROUPS.items():
                meta = CATEGORY_META.get(category, {"icon": "•", "desc": ""})
                with gr.Tab(f"{meta['icon']}  {category}"):
                    if meta.get("desc"):
                        gr.Markdown(f"<span class='auk-category-note'>{meta['desc']}</span>")
                    with gr.Tabs():
                        for task, task_examples in tasks.items():
                            examples = _resolve_demo_examples(task_examples)
                            if not examples:
                                continue
                            with gr.Tab(task):
                                is_instruct_tts = task == "Instruct TTS"
                                components = [gr.Textbox(label="Instruction", render=False)]
                                headers = ["Instruction"]
                                if not is_instruct_tts:
                                    components.insert(0, gr.Audio(label="Audio", render=False))
                                    headers.insert(0, "Audio")
                                dataset = gr.Dataset(
                                    components=components,
                                    samples=_demo_dataset_samples(task, examples),
                                    headers=headers,
                                    type="index",
                                    layout="table",
                                    samples_per_page=8,
                                    show_label=False,
                                    elem_classes="auk-examples-table",
                                )
                                dataset.select(
                                    lambda index, rows=examples: load_demo_example(index, rows),
                                    inputs=dataset,
                                    outputs=[in_audio, in_instr, in_secs, use_pe, in_seed],
                                    queue=False,
                                    api_visibility="private",
                                    show_progress="hidden",
                                )

    return demo


def main():
    p = argparse.ArgumentParser(description="AuK Gradio demo")
    p.add_argument(
        "--base_ckpt",
        default=None,
        help="AuK checkpoint. If any checkpoint flag is supplied, only explicitly supplied variants are shown.",
    )
    p.add_argument(
        "--flash_ckpt",
        default=None,
        help="AuK-Flash checkpoint. If any checkpoint flag is supplied, only explicitly supplied variants are shown.",
    )
    p.add_argument("--base_config", default=None)
    p.add_argument("--flash_config", default=None)
    p.add_argument("--base_device", default=None, help="Device for AuK, for example cuda:0.")
    p.add_argument("--flash_device", default=None, help="Device for AuK-Flash, for example cuda:1.")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--qwen_path", default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None, help="Data type. Auto-selected if omitted.")
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    explicitly_configured = args.base_ckpt is not None or args.flash_ckpt is not None
    candidates = {
        BASE_LABEL: (
            args.base_ckpt if explicitly_configured else default_path("ckpts", "AuK", "auk_base.safetensors"),
            args.base_config,
        ),
        FLASH_LABEL: (
            args.flash_ckpt if explicitly_configured else default_path("ckpts", "AuK-Flash", "auk_flash.safetensors"),
            args.flash_config,
        ),
    }
    for label, (checkpoint, config) in candidates.items():
        if checkpoint is None:
            continue
        if not os.path.isfile(checkpoint):
            if explicitly_configured:
                p.error(f"{label} checkpoint not found: {checkpoint}")
            continue
        if config is not None and not os.path.isfile(config):
            p.error(f"{label} config not found: {config}")
        CKPT_PATHS[label] = checkpoint
        CONFIG_PATHS[label] = config

    ENGINE_KWARGS.update(device=args.device, dtype=args.dtype, qwen_path=args.qwen_path)
    DEVICE_PATHS[BASE_LABEL] = args.base_device
    DEVICE_PATHS[FLASH_LABEL] = args.flash_device

    if not CKPT_PATHS:
        p.error("No valid --base_ckpt or --flash_ckpt was found.")
    if args.preload:
        for variant in CKPT_PATHS:
            get_engine(variant)

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=DEMO_CSS,
    )


if __name__ == "__main__":
    main()
