import os
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "custom_transcription_editor",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "build")
    _component_func = components.declare_component(
        "custom_transcription_editor", 
        path=build_dir
    )

def custom_transcription_editor(words_data, speaker_names=None, gap_threshold=0.6, search_query="", playback_rate=1.0, key=None):
    return _component_func(
        words_data=words_data,
        speaker_names=speaker_names or {},
        gap_threshold=gap_threshold,
        search_query=search_query,
        playback_rate=playback_rate,
        key=key,
        default=words_data
    )