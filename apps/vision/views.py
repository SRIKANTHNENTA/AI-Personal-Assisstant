from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from apps.emotion_tracker.models import EmotionState


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.log'}
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.rtf'}
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024


def _sentence_tokenize(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _extractive_summary(text: str, max_sentences: int = 4) -> str:
    cleaned = (text or '').strip()
    if not cleaned:
        return 'No readable text could be extracted.'

    sentences = _sentence_tokenize(cleaned)
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)

    words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned.lower())
    if not words:
        return ' '.join(sentences[:max_sentences])

    stop_words = {
        'the', 'and', 'for', 'that', 'with', 'this', 'from', 'your', 'have', 'will', 'into', 'were', 'been',
        'their', 'they', 'them', 'about', 'would', 'there', 'which', 'when', 'what', 'where', 'while', 'also',
        'are', 'was', 'you', 'our', 'its', 'can', 'not', 'has', 'had', 'but', 'all', 'any', 'out', 'one', 'two',
    }
    filtered = [w for w in words if w not in stop_words]
    if not filtered:
        return ' '.join(sentences[:max_sentences])

    freqs = Counter(filtered)
    sentence_scores: list[tuple[float, int, str]] = []
    for idx, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        if not sentence_words:
            continue
        score = sum(freqs.get(w, 0) for w in sentence_words) / max(1, len(sentence_words))
        sentence_scores.append((score, idx, sentence))

    top = sorted(sentence_scores, key=lambda item: item[0], reverse=True)[:max_sentences]
    ordered = [item[2] for item in sorted(top, key=lambda item: item[1])]
    return ' '.join(ordered)


def _key_points(text: str, max_points: int = 5) -> list[str]:
    sentences = _sentence_tokenize(text)
    points: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip(' -\t\n\r')
        if len(stripped) >= 20:
            points.append(stripped)
        if len(points) >= max_points:
            break
    return points


def _top_keywords(text: str, max_keywords: int = 8) -> list[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', (text or '').lower())
    if not words:
        return []
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'were', 'been', 'which', 'about', 'into', 'their',
        'there', 'would', 'could', 'should', 'after', 'before', 'while', 'where', 'when', 'your',
        'image', 'document', 'text', 'using', 'scan', 'scanned',
    }
    filtered = [w for w in words if w not in stop_words]
    if not filtered:
        return []
    return [w for w, _ in Counter(filtered).most_common(max_keywords)]


def _build_xai_block(
    statement: str,
    justification: str,
    rules: list[str],
    evidence: list[str],
    confidence: float,
    mode: str,
    technical_notes: list[str] | None = None,
) -> dict:
    return {
        'mode': mode,
        'statement': statement,
        'justification': justification,
        'rules': rules,
        'evidence': evidence,
        'confidence': round(max(0.0, min(1.0, confidence)), 2),
        'technical_notes': technical_notes or [],
    }


def _extract_text_from_document(upload) -> tuple[str, str]:
    ext = Path(upload.name).suffix.lower()
    raw = upload.read()
    upload.seek(0)

    if ext in {'.txt', '.md', '.log', '.rtf'}:
        return raw.decode('utf-8', errors='ignore'), 'utf-8 decode'

    if ext == '.csv':
        text = raw.decode('utf-8', errors='ignore')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        preview = '\n'.join(lines[:120])
        return preview, 'csv text extract'

    if ext == '.json':
        parsed = json.loads(raw.decode('utf-8', errors='ignore'))
        pretty = json.dumps(parsed, indent=2)
        return pretty, 'json parse'

    if ext == '.pdf':
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return '', 'PDF extraction unavailable (install pypdf)'
        reader = PdfReader(io.BytesIO(raw))
        chunks: list[str] = []
        for page in reader.pages[:30]:
            page_text = page.extract_text() or ''
            if page_text.strip():
                chunks.append(page_text)
        return '\n'.join(chunks), 'pypdf'

    if ext == '.docx':
        try:
            from docx import Document  # type: ignore
        except Exception:
            return '', 'DOCX extraction unavailable (install python-docx)'
        document = Document(io.BytesIO(raw))
        lines = [p.text for p in document.paragraphs if p.text.strip()]
        return '\n'.join(lines), 'python-docx'

    return '', 'Unsupported document format'


def _extract_image_signals(upload) -> tuple[str, str, str, str, dict]:
    """
    Returns: caption, ocr_text, object_label, emotion_label
    """
    ext = Path(upload.name).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        return '', '', 'Unknown', 'Neutral', {}

    raw = upload.read()
    upload.seek(0)

    caption = 'Image uploaded and decoded successfully.'
    ocr_text = ''
    object_label = 'Detected'
    emotion_label = 'Neutral'
    diagnostics: dict[str, str] = {}

    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        image = Image.open(io.BytesIO(raw)).convert('RGB')
        np_image = np.array(image)

        # Basic visual descriptor fallback that works without remote models.
        height, width = np_image.shape[:2]
        brightness = float(np.mean(np_image))
        tone = 'bright' if brightness >= 140 else 'dim'
        orientation = 'landscape' if width >= height else 'portrait'
        caption = f'{tone.title()} {orientation} image ({width}x{height}) with {"high" if brightness >= 170 else "moderate" if brightness >= 120 else "low"} lighting.'
        diagnostics['dimensions'] = f'{width}x{height}'
        diagnostics['brightness'] = f'{brightness:.1f}'
        diagnostics['tone'] = tone
        diagnostics['orientation'] = orientation

        try:
            import cv2  # type: ignore
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = float(edges.mean())
            if edge_density > 24:
                object_label = 'Complex scene'
            elif edge_density > 9:
                object_label = 'Single object likely'
            else:
                object_label = 'Document/flat image'
            diagnostics['edge_density'] = f'{edge_density:.2f}'
        except Exception:
            object_label = 'Detected'

        # OCR with optional pytesseract.
        try:
            import pytesseract  # type: ignore
            text = pytesseract.image_to_string(image)
            ocr_text = text.strip()
            diagnostics['ocr_chars'] = str(len(ocr_text))
        except Exception:
            ocr_text = ''

        # Optional emotion cue from existing multimodal detector.
        try:
            from ai_services.multimodal_emotion_detector import MultimodalEmotionDetector

            detector = MultimodalEmotionDetector()
            result = detector.detect_multimodal_emotion(image_data=np_image)
            fused = result.get('fused_emotion') if isinstance(result, dict) else None
            if isinstance(fused, dict):
                emotion_label = str(fused.get('emotion', 'Neutral')).title()
        except Exception:
            emotion_label = 'Neutral'

    except Exception:
        caption = 'Unable to decode image content. Please upload a valid image.'
        object_label = 'Unknown'

    return caption, ocr_text, object_label, emotion_label, diagnostics


@login_required
def vision_home(request):
    upload = request.FILES.get('upload_file') if request.method == 'POST' else None
    scan_result = None
    message = None
    explanation_level = (request.POST.get('explanation_level') or 'detailed').strip().lower()
    if explanation_level not in {'basic', 'detailed', 'technical'}:
        explanation_level = 'detailed'

    if upload:
        if upload.size > MAX_UPLOAD_SIZE_BYTES:
            message = 'File is too large. Please upload files up to 15 MB.'
        else:
            ext = Path(upload.name).suffix.lower()
            content_type = getattr(upload, 'content_type', 'application/octet-stream')
            ocr_text = ''
            caption = ''
            extracted_text = ''
            extraction_mode = ''

            if ext in IMAGE_EXTENSIONS or content_type.startswith('image/'):
                caption, ocr_text, obj_label, emotion_label, diagnostics = _extract_image_signals(upload)
                text_source = ocr_text if ocr_text else caption
                summary = _extractive_summary(text_source)
                points = _key_points(text_source)
                ocr_words = len(re.findall(r'\b\w+\b', ocr_text)) if ocr_text else 0
                evidence = [
                    f"Dimensions: {diagnostics.get('dimensions', 'unknown')}",
                    f"Brightness score: {diagnostics.get('brightness', 'n/a')}",
                    f"Scene complexity: {obj_label}",
                    f"OCR words detected: {ocr_words}",
                ]
                if diagnostics.get('edge_density'):
                    evidence.append(f"Edge density: {diagnostics.get('edge_density')}")

                confidence = 0.62
                if obj_label == 'Complex scene':
                    confidence += 0.1
                if ocr_words > 8:
                    confidence += 0.15
                if diagnostics.get('dimensions'):
                    confidence += 0.05

                xai = _build_xai_block(
                    statement=caption,
                    justification=(
                        'Caption is generated from visual signals (brightness, orientation, structure) and '
                        'enhanced with OCR evidence when text is present.'
                    ),
                    rules=[
                        'Use image geometry and brightness to describe scene style.',
                        'Use edge density to estimate visual complexity.',
                        'If OCR text exists, prioritize text-aware summary.',
                    ],
                    evidence=evidence,
                    confidence=confidence,
                    mode='Image Captioning + OCR',
                    technical_notes=[
                        f"Brightness mean (pixel): {diagnostics.get('brightness', 'n/a')}",
                        f"Edge density metric: {diagnostics.get('edge_density', 'n/a')}",
                        f"Orientation heuristic: {diagnostics.get('orientation', 'n/a')}",
                    ],
                )
                scan_result = {
                    'file_name': upload.name,
                    'content_type': content_type,
                    'caption': caption,
                    'summary': summary,
                    'key_points': points,
                    'xai': xai,
                    'ocr_preview': (ocr_text[:1200] + '...') if len(ocr_text) > 1200 else ocr_text,
                    'profile': [
                        {'label': 'Object', 'value': obj_label},
                        {'label': 'OCR', 'value': 'Detected' if ocr_text else 'Unavailable'},
                        {'label': 'Emotion', 'value': emotion_label},
                    ],
                    'analysis_type': 'image',
                }
            elif ext in TEXT_EXTENSIONS or ext in DOCUMENT_EXTENSIONS or content_type.startswith('text/'):
                extracted_text, extraction_mode = _extract_text_from_document(upload)
                if extracted_text.strip():
                    summary = _extractive_summary(extracted_text)
                    points = _key_points(extracted_text)
                    keywords = _top_keywords(extracted_text)
                    sentence_count = len(_sentence_tokenize(extracted_text))
                    evidence = [
                        f"Extraction method: {extraction_mode}",
                        f"Characters analyzed: {len(extracted_text)}",
                        f"Sentences detected: {sentence_count}",
                    ]
                    if keywords:
                        evidence.append('Top keywords: ' + ', '.join(keywords[:6]))

                    confidence = 0.65
                    if len(extracted_text) > 400:
                        confidence += 0.1
                    if sentence_count >= 4:
                        confidence += 0.1
                    if keywords:
                        confidence += 0.05

                    xai = _build_xai_block(
                        statement=f"Document content analyzed with {extraction_mode} and summarized from extracted text evidence.",
                        justification='Summary and key points are generated from the highest-information sentences and keyword frequency.',
                        rules=[
                            'Extract readable text from file format parser.',
                            'Rank informative sentences by term frequency.',
                            'Expose top keywords to explain summary focus.',
                        ],
                        evidence=evidence,
                        confidence=confidence,
                        mode='Document Understanding (Explainable)',
                        technical_notes=[
                            f"Sentence tokenizer count: {sentence_count}",
                            f"Keyword model: frequency-based ranking",
                            f"Extraction backend: {extraction_mode}",
                        ],
                    )
                    scan_result = {
                        'file_name': upload.name,
                        'content_type': content_type,
                        'caption': f'Document scanned using {extraction_mode}.',
                        'summary': summary,
                        'key_points': points,
                        'xai': xai,
                        'ocr_preview': (extracted_text[:1500] + '...') if len(extracted_text) > 1500 else extracted_text,
                        'profile': [
                            {'label': 'Object', 'value': 'Document'},
                            {'label': 'OCR', 'value': 'Parsed text'},
                            {'label': 'Emotion', 'value': 'N/A'},
                        ],
                        'analysis_type': 'document',
                    }
                else:
                    message = f'Could not read document text. {extraction_mode}'
            else:
                message = 'Unsupported file type. Upload images, txt, csv, json, pdf, or docx files.'

    recent_emotion = EmotionState.objects.filter(user=request.user).order_by('-detected_at').first()

    return render(request, 'vision/index.html', {
        'scan_result': scan_result,
        'scan_message': message,
        'explanation_level': explanation_level,
        'recent_emotion': recent_emotion,
        'camera_states': [
            'Camera ready',
            'Live feed panel',
            'Neon scan active',
        ],
    })
