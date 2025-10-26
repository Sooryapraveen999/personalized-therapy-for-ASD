import os
import json
import pathlib
from collections import Counter

from emotion_detection_system.conf import emotion_detection_system_folder, DATA_EXPERIMENT_SLUG


THERAPY_SUGGESTIONS = {
    'green': {
        'summary': 'Calm/regulated (low arousal, positive valence).',
        'suggestions': [
            'Proceed with skill-building tasks and structured learning.',
            'Use social engagement, joint attention, and reinforcement.',
            'Introduce communication practice and turn-taking activities.'
        ]
    },
    'yellow': {
        'summary': 'Excited/over-aroused (high arousal, positive valence).',
        'suggestions': [
            'Channel energy into movement-based or sensory activities.',
            'Use clear transitions, timers, and brief task segments.',
            'Practice regulation strategies (paced breathing, grounding).' 
        ]
    },
    'red': {
        'summary': 'Anger/stress/anxiety (high arousal, negative valence).',
        'suggestions': [
            'Decrease stimulation; offer quiet space or sensory breaks.',
            'Coach emotion labeling and simple coping strategies.',
            'Use predictable routines and reduce task demands temporarily.'
        ]
    },
    'blue': {
        'summary': 'Low energy/sad/withdrawn (low arousal, negative valence).',
        'suggestions': [
            'Increase engagement with preferred or highly motivating items.',
            'Use short success-oriented tasks to build momentum.',
            'Incorporate joint play and positive affect to elevate mood.'
        ]
    }
}


def _get_test_dataframe(classifier):
    if classifier.configuration.is_multimodal and classifier.configuration.fusion_type == 'late_fusion':
        return classifier.dataset.x_test_video
    if 'video' in classifier.configuration.modalities:
        return classifier.dataset.x_test_video
    if 'audio' in classifier.configuration.modalities:
        return classifier.dataset.x_test_audio
    return None


def _build_timeline(classifier, test_df):
    idx = list(classifier._prediction_probabilities.index) if classifier._prediction_probabilities is not None else list(classifier.dataset.y_test.index)
    preds = list(classifier._prediction_labels)
    rows = []
    for i, ridx in enumerate(idx):
        row = {'index': int(ridx), 'emotion': preds[i]}
        if test_df is not None:
            if 'time_of_video_seconds' in test_df.columns:
                val = test_df.loc[ridx, 'time_of_video_seconds']
                try:
                    row['time_of_video_seconds'] = float(val)
                except Exception:
                    row['time_of_video_seconds'] = None
            if 'frametime' in test_df.columns:
                row['frametime'] = str(test_df.loc[ridx, 'frametime'])
            if 'video_part' in test_df.columns:
                row['video_part'] = str(test_df.loc[ridx, 'video_part'])
        rows.append(row)
    def _sort_key(r):
        if 'time_of_video_seconds' in r and r['time_of_video_seconds'] is not None:
            return (r.get('video_part', ''), r['time_of_video_seconds'])
        return (r.get('video_part', ''), r.get('frametime', ''))
    rows.sort(key=_sort_key)
    segments = []
    if not rows:
        return rows, segments
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]['emotion'] != rows[start]['emotion']:
            seg = rows[start:i]
            segment = {
                'emotion': rows[start]['emotion'],
                'start_index': seg[0]['index'],
                'end_index': seg[-1]['index'],
                'length': len(seg)
            }
            st = seg[0].get('time_of_video_seconds')
            et = seg[-1].get('time_of_video_seconds')
            if st is not None and et is not None:
                segment['start_time_seconds'] = float(st)
                segment['end_time_seconds'] = float(et)
            segment['start_frametime'] = seg[0].get('frametime')
            segment['end_frametime'] = seg[-1].get('frametime')
            segments.append(segment)
            start = i
    segments_sorted = sorted(segments, key=lambda s: s['length'], reverse=True)
    return rows, segments_sorted[:3]


essential_emotions = ['blue', 'green', 'red', 'yellow']


def generate_therapy_form(classifier, path_json, folder_to_save_name):
    test_df = _get_test_dataframe(classifier)
    timeline, segments_top = _build_timeline(classifier, test_df)
    counts = Counter([r['emotion'] for r in timeline])
    distribution = {e: int(counts.get(e, 0)) for e in essential_emotions}
    primary = max(distribution, key=lambda k: distribution[k]) if timeline else None

    therapy = {
        'experiment_slug': DATA_EXPERIMENT_SLUG,
        'config_slug': pathlib.Path(path_json).name.split('.json')[0],
        'participant': int(classifier.configuration.participant_number),
        'session': str(classifier.configuration.session_number),
        'annotation_type': classifier.configuration.annotation_type,
        'metrics': {
            'accuracy': float(classifier.accuracy) if classifier.accuracy is not None else None,
            'balanced_accuracy': float(classifier.balanced_accuracy) if classifier.balanced_accuracy is not None else None
        },
        'distribution': distribution,
        'segments_top3': segments_top,
        'recommendations': {
            'primary_emotion': primary,
            'primary_summary': THERAPY_SUGGESTIONS.get(primary, {}).get('summary') if primary else None,
            'primary_interventions': THERAPY_SUGGESTIONS.get(primary, {}).get('suggestions') if primary else None,
            'by_emotion': {
                e: THERAPY_SUGGESTIONS[e] for e in essential_emotions if distribution.get(e, 0) > 0
            }
        }
    }

    out_dir = os.path.join(emotion_detection_system_folder, 'therapy_reports', folder_to_save_name, classifier.configuration.annotation_type)
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"{DATA_EXPERIMENT_SLUG}_{therapy['config_slug']}"
    json_path = os.path.join(out_dir, base_name + '_therapy.json')
    text_path = os.path.join(out_dir, base_name + '_therapy.txt')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(therapy, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"Therapy Plan for ASD Session")
    lines.append(f"Participant: {therapy['participant']}  Session: {therapy['session']}  Annotation: {therapy['annotation_type']}")
    lines.append(f"Experiment: {therapy['experiment_slug']}  Config: {therapy['config_slug']}")
    lines.append("")
    lines.append(f"Metrics: Accuracy={therapy['metrics']['accuracy']}, Balanced Accuracy={therapy['metrics']['balanced_accuracy']}")
    lines.append("Distribution of predicted emotion zones:")
    for e in essential_emotions:
        lines.append(f"- {e}: {distribution.get(e, 0)}")
    if primary:
        lines.append("")
        lines.append(f"Primary zone observed: {primary}")
        pr = THERAPY_SUGGESTIONS.get(primary, {})
        if pr.get('summary'):
            lines.append(f"Summary: {pr['summary']}")
        if pr.get('suggestions'):
            lines.append("Suggested interventions:")
            for s in pr['suggestions']:
                lines.append(f"- {s}")
    if segments_top:
        lines.append("")
        lines.append("Top segments:")
        for seg in segments_top:
            st = seg.get('start_time_seconds')
            et = seg.get('end_time_seconds')
            if st is not None and et is not None:
                lines.append(f"- {seg['emotion']} | {st:.2f}s → {et:.2f}s ({seg['length']} samples)")
            else:
                lines.append(f"- {seg['emotion']} | idx {seg['start_index']} → {seg['end_index']} ({seg['length']} samples)")

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    return json_path, text_path


def build_therapy_paragraph(classifier, path_json):
    test_df = _get_test_dataframe(classifier)
    timeline, segments_top = _build_timeline(classifier, test_df)
    counts = Counter([r['emotion'] for r in timeline])
    distribution = {e: int(counts.get(e, 0)) for e in essential_emotions}
    primary = max(distribution, key=lambda k: distribution[k]) if timeline else None

    part = int(classifier.configuration.participant_number)
    sess = str(classifier.configuration.session_number)
    acc = classifier.accuracy if classifier.accuracy is not None else 0.0
    bacc = classifier.balanced_accuracy if classifier.balanced_accuracy is not None else 0.0
    primary_sum = THERAPY_SUGGESTIONS.get(primary, {}).get('summary') if primary else None
    suggs = THERAPY_SUGGESTIONS.get(primary, {}).get('suggestions') if primary else []

    dist_txt = ", ".join([f"{e}: {distribution[e]}" for e in essential_emotions])
    seg_txt = ""
    if segments_top:
        formatted = []
        for seg in segments_top:
            st = seg.get('start_time_seconds')
            et = seg.get('end_time_seconds')
            if st is not None and et is not None:
                formatted.append(f"{seg['emotion']} from {st:.2f}s to {et:.2f}s")
            else:
                formatted.append(f"{seg['emotion']} (idx {seg['start_index']}→{seg['end_index']})")
        seg_txt = "; ".join(formatted)

    sugg_txt = "; ".join(suggs) if suggs else ""

    paragraph = (
        f"For participant {part}, session {sess}, the model achieved Accuracy={acc:.2f} and "
        f"Balanced Accuracy={bacc:.2f}. The predicted emotion distribution was {dist_txt}. "
        f"The primary observed zone was {primary}. {primary_sum if primary_sum else ''} "
        f"Recommended interventions: {sugg_txt}. "
        f"Key segments observed: {seg_txt}."
    )
    return paragraph

def build_therapy_bulleted_summary(classifier, path_json):
    test_df = _get_test_dataframe(classifier)
    timeline, segments_top = _build_timeline(classifier, test_df)
    counts = Counter([r['emotion'] for r in timeline])
    distribution = {e: int(counts.get(e, 0)) for e in essential_emotions}
    primary = max(distribution, key=lambda k: distribution[k]) if timeline else None

    part = int(classifier.configuration.participant_number)
    sess = str(classifier.configuration.session_number)
    acc = classifier.accuracy if classifier.accuracy is not None else 0.0
    bacc = classifier.balanced_accuracy if classifier.balanced_accuracy is not None else 0.0
    pr = THERAPY_SUGGESTIONS.get(primary, {}) if primary else {}

    lines = []
    lines.append(f"Participant: {part}")
    lines.append(f"Session: {sess}")
    lines.append(f"Annotation: {classifier.configuration.annotation_type}")
    lines.append(f"Accuracy: {acc:.2f}")
    lines.append(f"Balanced Accuracy: {bacc:.2f}")
    if primary:
        lines.append(f"Primary Zone: {primary}")
        if pr.get('summary'):
            lines.append(f"Summary: {pr['summary']}")
        if pr.get('suggestions'):
            lines.append("Interventions:")
            for s in pr['suggestions']:
                lines.append(f"- {s}")
    lines.append("Distribution:")
    for e in essential_emotions:
        lines.append(f"- {e}: {distribution.get(e, 0)}")
    if segments_top:
        lines.append("Top Segments:")
        for seg in segments_top:
            st = seg.get('start_time_seconds')
            et = seg.get('end_time_seconds')
            if st is not None and et is not None:
                lines.append(f"- {seg['emotion']} | {st:.2f}s to {et:.2f}s ({seg['length']} samples)")
            else:
                lines.append(f"- {seg['emotion']} | idx {seg['start_index']} to {seg['end_index']} ({seg['length']} samples)")
    return "\n".join(lines)
