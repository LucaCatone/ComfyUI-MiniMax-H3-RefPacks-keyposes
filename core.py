"""Reference sets, deterministic addressing and native H3 payload assembly."""
import math
import json
import re

import torch

TARGETS = ("h3", "h3_video", "h3_image")
from bexplains_refpacks.core import FORMAT, VERSION, add_pack, empty_set, subject_numbers


def model_target(model):
    base = getattr(model, "model", None)
    names = {cls.__name__ for cls in type(base).__mro__}
    if "PH3Image" in names:
        raise ValueError("p-h3-image is not supported by this release. Use native MiniMax H3 Image or H3 Ref2VA.")
    if "MiniMaxH3Image" in names:
        return "h3_image"
    if "MiniMaxH3" in names:
        return "h3_video"
    raise ValueError("The MODEL input must be a native ComfyUI MiniMax H3 model. For video select a Ref2VA checkpoint.")


def validate_latent(z, kind):
    if not isinstance(z, torch.Tensor) or z.is_nested:
        raise ValueError("Expected a plain H3 reference tensor, not a joint generation latent.")
    if z.ndim == 4:
        z = z.unsqueeze(2)
    if z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 24:
        raise ValueError(f"Expected H3 reference [1,24,T,H,W], got {tuple(z.shape)}. Use the correct H3 VAE.")
    if not z.is_floating_point() or not torch.isfinite(z).all():
        raise ValueError("Reference latents must contain finite floating-point values.")
    if min(z.shape[2:]) < 1:
        raise ValueError("Reference latent dimensions must be positive.")
    if kind == "image" and z.shape[2] != 1:
        raise ValueError("Each image reference must have one latent time step. Use separate entries for multiple images.")
    return z


def validate_pack(pack, require_vision=True):
    if not isinstance(pack, dict) or pack.get("format") != FORMAT or pack.get("version") != VERSION:
        raise ValueError("Not a supported BExplains RefPack.")
    if pack.get("target") not in TARGETS:
        raise ValueError("Unsupported RefPack model family.")
    entries = pack.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A RefPack must contain at least one reference.")
    for e in entries:
        if e.get("kind") == "audio":
            validate_audio_latent(e.get("audio_latent"))
            if e.get("ref_audio_t") != e["audio_latent"].shape[-1]:
                raise ValueError("Audio reference length does not match its latent.")
            if pack["target"] != "h3_video":
                raise ValueError("Audio references require H3 Ref2VA.")
            continue
        if e.get("kind") not in ("image", "video"):
            raise ValueError("Unsupported reference kind.")
        validate_latent(e.get("latent"), e["kind"])
        if pack["target"] == "h3_image" and e["kind"] == "video":
            raise ValueError("H3 Image does not accept video RefPacks.")
        v = e.get("vision")
        if require_vision and v is None:
            raise ValueError("This pack lacks H3 text-encoder vision inputs. Recreate it with the reference encoder or supply matching vision images when importing a latent.")
        if v is not None:
            if not isinstance(v, torch.Tensor) or v.ndim != 4 or v.shape[-1] != 3 or min(v.shape[:3]) < 1:
                raise ValueError("Vision data must be RGB frames [N,H,W,3].")
            if not v.is_floating_point() or not torch.isfinite(v).all() or v.min() < 0 or v.max() > 1:
                raise ValueError("Vision inputs must be finite floating-point RGB frames in [0,1].")
            if e["kind"] == "image" and v.shape[0] != 1:
                raise ValueError("Each image entry requires exactly one vision image.")
            if e["kind"] == "video":
                ts = e.get("timestamps", [])
                if len(ts) != v.shape[0] or any(not math.isfinite(float(t)) for t in ts):
                    raise ValueError("Each video vision frame needs a finite timestamp.")
                if any(float(b) < float(a) for a, b in zip(ts, ts[1:])):
                    raise ValueError("Video timestamps must be in chronological order.")
    return pack


def validate_audio_latent(z):
    if not isinstance(z, torch.Tensor) or z.is_nested or z.ndim != 4 or tuple(z.shape[:3]) != (1,32,2) or z.shape[-1] < 1:
        raise ValueError("Expected H3 audio reference latent [1,32,2,T]. Use the H3 audio VAE.")
    if not z.is_floating_point() or not torch.isfinite(z).all():
        raise ValueError("Audio latents must contain finite floating-point values.")
    return z


def reference_window(references, output_frames, compress_motion=False):
    """Native leading-window policy by default; full references are opt-in.

    H3's causal video encoder permits slicing a cached latent prefix on the
    native 17n+5 frame grid. Vision frames use the same source-time cutoff.
    Stored files and the caller's reference set are never modified.
    """
    if compress_motion:
        return references
    if output_frames < 5 or output_frames % 17 != 5:
        raise ValueError("Reference window requires an H3-aligned output frame count.")
    groups = []
    for group in references.get("groups", []):
        entries = []
        for entry in group["entries"]:
            if entry["kind"] != "video":
                entries.append(entry)
                continue
            z = validate_latent(entry["latent"], "video")
            if z.shape[2] < 2 or (z.shape[2] - 2) % 5:
                raise ValueError("Reference video does not have a native H3 temporal grid. Recreate it with Encode H3 Video Refpack.")
            source_frames = int(entry.get("frame_count", ((z.shape[2]-2)//5)*17+5))
            expected_t = 2 + ((source_frames-5)//17)*5
            if source_frames < 5 or source_frames % 17 != 5 or expected_t != z.shape[2]:
                raise ValueError("Reference frame count and H3 latent timeline disagree. Recreate this reference pack.")
            if source_frames <= output_frames:
                entries.append(entry)
                continue
            kept_t = 2 + ((output_frames-5)//17)*5
            kept = [i for i,t in enumerate(entry["timestamps"]) if float(t) < output_frames/24.0]
            if not kept:
                raise ValueError("Reference video has no vision frames inside the requested leading window.")
            cropped = dict(entry)
            cropped.update(latent=z[:, :, :kept_t].contiguous(),
                           vision=entry["vision"][kept].contiguous(),
                           timestamps=[entry["timestamps"][i] for i in kept],
                           source_frame_count=source_frames, frame_count=output_frames)
            entries.append(cropped)
        groups.append({**group, "entries": entries})
    return {**references, "groups": groups}


def reference_layout(references):
    """One ordering/numbering implementation for inspection and conditioning."""
    if not references or not references.get("groups"):
        raise ValueError("Add at least one reference pack.")
    counters = {"image":0, "video":0, "audio":0}
    for kind in ("image", "video", "audio"):
        for gi,g in enumerate(references["groups"]):
            for ei,e in enumerate(g["entries"]):
                if e["kind"] == kind:
                    counters[kind] += 1
                    label = f"<{dict(image='Picture',video='Video',audio='Audio')[kind]} {counters[kind]}>"
                    yield gi,ei,e,label


RAW_PROMPT_SECTIONS = frozenset((
    "subject_definitions", "summary", "retention_analysis", "detailed_description",
    "description", "overall_soundscape", "non_diegetic_music",
))
RAW_PROMPT_HEADER = re.compile(
    r"^[ \t]*(?:" + "|".join(sorted(RAW_PROMPT_SECTIONS)) + r")[ \t]*:",
    re.MULTILINE | re.IGNORECASE,
)


def is_raw_prompt(prompt):
    """Recognize H3 section headers or a JSON object with H3 section keys."""
    if RAW_PROMPT_HEADER.search(prompt):
        return True
    try:
        value = json.loads(prompt)
    except (ValueError, TypeError):
        return False
    return isinstance(value, dict) and bool(RAW_PROMPT_SECTIONS.intersection(value))


def _timecode(frame):
    """mm:ss.mmm at 24 fps, for beat timestamps in the compiled prompt."""
    seconds = int(round(frame)) / 24.0
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def assemble(references, prompt, task, auto_outfit=True, allow_raw_prompt=False, shot_frames=None, video_vision=True):
    ordered = list(reference_layout(references))
    if task not in ("h3_image", "h3_video"):
        raise ValueError("Unsupported receiving model family.")
    if task == "h3_image" and any(e["kind"] != "image" for g in references["groups"] for e in g["entries"]):
        raise ValueError("H3 Image accepts still-image references only. Remove motion/video and audio packs; still-image packs work with either H3 model.")
    # H3's native Ref2VA presentation is images, then videos, then audio.
    labels, counters, blocks, items = {}, {"image": 0, "video": 0, "audio": 0}, [], []
    for gi, ei, e, label in ordered:
        kind = e["kind"]
        counters[kind] += 1
        labels[(gi, ei)] = label
        if kind == "audio":
            z = validate_audio_latent(e["audio_latent"])
            blocks.append({"kind":"audio", "ref_audio_t":z.shape[-1], "audio_latent":z})
            items.append({"type":"audio"})
            continue
        z = validate_latent(e["latent"], kind)
        block = {"kind": kind, "latent_h": z.shape[-2], "latent_w": z.shape[-1], "latent": z}
        item = {"type": kind, "data": e["vision"]}
        if kind == "video":
            block.update(latent_t=z.shape[2], ref_audio_t=0, audio_latent=None)
            if video_vision:
                item["timestamps"] = list(e["timestamps"])
            else:
                item["data"] = e["vision"][:1]
                item["timestamps"] = list(e["timestamps"])[:1]
        blocks.append(block)
        items.append(item)
    definitions, outfit_aliases, location_aliases, report, subject_ids = [], {}, {}, [], set()
    typed_aliases, item_ids, wearer_ids, character_ids = {}, set(), set(), set()
    keypose_labels = []
    keypose_frames = []
    subjects = subject_numbers(references)
    outfit_labels = set()
    presentation_order = sorted(enumerate(references["groups"]), key=lambda pair: ({"subject": 0, "character": 0, "item": 0, "outfit": 1, "location": 2, "style": 3, "reference": 3, "video": 4, "motion": 4, "keyposes": 4, "audio":5}[pair[1]["role"]], subjects.get(pair[0], pair[1]["role_id"])))
    for gi, g in presentation_order:
        ls = [labels[(gi, ei)] for ei in range(len(g["entries"]))]
        report.append(f"{g['role']} {g['role_id']}: {g['name']} -> {', '.join(ls)}")
        if g["role"] in ("subject", "character", "item"):
            sid = subjects[gi]
            subject_ids.add(sid)
            if g["role"] == "character":
                alias = f"<Character {g['role_id']}>"
                typed_aliases[alias] = f"<Subject {sid}>"
                report[-1] = f"{alias} -> <Subject {sid}>: {g['name']} -> {', '.join(ls)}"
                wearer_ids.add(sid)
                character_ids.add(sid)
                definitions.append(f"<Subject {sid}> is one person whose identity, facial features, hairstyle and body proportions are shown in {' and '.join(ls)}. These pictures show the same person.")
            elif g["role"] == "item":
                alias = f"<Item {g['role_id']}>"
                typed_aliases[alias] = f"<Subject {sid}>"
                report[-1] = f"{alias} -> <Subject {sid}>: {g['name']} -> {', '.join(ls)}"
                item_ids.add(sid)
                definitions.append(f"<Subject {sid}> is one object shown in {' and '.join(ls)}. These pictures show the same item; preserve its shape, proportions, colors, materials and distinctive details.")
            else:
                wearer_ids.add(sid)
                definitions.append(f"<Subject {sid}> is one subject shown in {' and '.join(ls)}. These pictures show the same subject; preserve its defining visual features.")
        elif g["role"] == "outfit":
            alias = f"<Outfit {g['role_id']}>"
            outfit_aliases[alias] = " and ".join(ls)
            outfit_labels.update(ls)
            report[-1] = f"{alias}: {g['name']} -> {', '.join(ls)}"
            definitions.append(f"{' and '.join(ls)} supplies the same outfit reference; it does not define an additional person.")
        elif g["role"] == "location":
            alias = f"<Location {g['role_id']}>"
            location_aliases[alias] = "the location shown in " + " and ".join(ls)
            report[-1] = f"{alias}: {g['name']} -> {', '.join(ls)}"
            definitions.append(f"{' and '.join(ls)} defines the same location, environment and architecture.")
        elif g["role"] == "style":
            alias = f"<Style {g['role_id']}>"
            typed_aliases[alias] = " and ".join(ls)
            report[-1] = f"{alias}: {g['name']} -> {', '.join(ls)}"
            definitions.append(f"{' and '.join(ls)} supplies the visual style reference; use its aesthetic, colors, lighting and rendering style as requested, without defining an additional subject.")
        elif g["role"] == "keyposes":
            alias = f"<Poses {g['role_id']}>"
            typed_aliases[alias] = " and ".join(ls)
            report[-1] = f"{alias}: {g['name']} -> {', '.join(ls)}"
            keypose_labels.extend(ls)
            if all(e["kind"] == "image" for e in g["entries"]):
                keypose_frames.extend(ls)
                definitions.append(f"{' and '.join(ls)} are consecutive key frames of the same shot, in this order. Each picture is one moment of one continuous take: treat them as frames of the target video, never as portraits, separate subjects or separate shots.")
            elif not any(e["kind"] == "image" for e in g["entries"]):
                for label in ls:
                    definitions.append(f"{label} is a key-pose reference, not footage: it shows the key poses of the shot in playback order, each held at its moment and separated by cuts, using colored stand-ins. The stand-ins' positions, left-right order, occlusion and camera height must be followed exactly; the motion between the poses is generated smoothly. The stand-ins' colors and surfaces are replaced by the referenced characters; each distinct colored stand-in is exactly one person. It is never shown in the target video.")
        elif g["role"] in ("video", "motion"):
            if g["role"] == "motion":
                alias = f"<Motion {g['role_id']}>"
                typed_aliases[alias] = " and ".join(ls)
                report[-1] = f"{alias}: {g['name']} -> {', '.join(ls)}"
            for label in ls:
                definitions.append(f"{label} supplies the requested motion, poses, movement and temporal structure for the target subject." if g["role"] == "motion" else f"{label} supplies the source performance, movement and temporal structure.")
        elif g["role"] == "audio":
            for label in ls:
                definitions.append(f"{label} supplies an audio reference; use it as requested in the shot instruction.")
    if not prompt.strip():
        raise ValueError("Enter a shot prompt.")
    if is_raw_prompt(prompt):
        if not allow_raw_prompt:
            raise ValueError("Structured H3 prompt detected. Enable Allow Raw Prompt to pass it through unchanged, or enter a shot instruction.")
        return prompt, blocks, items, "\n".join(report)
    text = prompt.strip()
    if "<Picture N>" in text:
        if len(outfit_aliases) != 1:
            raise ValueError("<Picture N> requires exactly one outfit group. Use <Outfit 1>, <Outfit 2>, etc. for multiple outfits.")
        text = text.replace("<Picture N>", next(iter(outfit_aliases.values())))
    explicit_outfit = any(alias in text for alias in outfit_aliases) or any(label in text for label in outfit_labels)
    for alias, actual in {**typed_aliases, **outfit_aliases, **location_aliases}.items():
        text = text.replace(alias, actual)
    unknown_role = re.search(r"<(Character|Item|Motion|Poses|Outfit|Location|Style) \d+>", text)
    if unknown_role:
        raise ValueError(f"Prompt refers to missing {unknown_role.group(0)}. Check the reference map.")
    if auto_outfit and len(outfit_aliases) == 1 and len(wearer_ids) == 1 and not explicit_outfit:
        if item_ids:
            text = text.rstrip() + f" <Subject {next(iter(wearer_ids))}> is wearing the outfit in {next(iter(outfit_aliases.values()))}."
        else:
            text = text.rstrip(" .") + f", wearing the outfit in {next(iter(outfit_aliases.values()))}."
    all_labels = set(labels.values())
    for kind, number in re.findall(r"<(Picture|Video|Audio|Subject) (\d+)>", text):
        if (kind == "Subject" and int(number) not in subject_ids) or (kind != "Subject" and f"<{kind} {number}>" not in all_labels):
            raise ValueError(f"Prompt refers to missing <{kind} {number}>. Check the reference map.")
    if task == "h3_video":
        soundscape = ("Follow the shot instruction for referenced audio, speech and music." if counters['audio'] else "No dialogue.")
        music = ("Use referenced music only as requested in the shot instruction." if counters['audio'] else "No music is requested.")
        subject_retention = "\n".join(
            f"<Subject {i}>: preserve object identity, shape, proportions, colors, materials and distinctive details."
            if i in item_ids else
            f"<Subject {i}>: preserve identity, face, hair and body proportions; apply an explicitly referenced outfit where requested."
            if i in character_ids else
            f"<Subject {i}>: preserve defining visual features; apply an explicitly referenced outfit where requested."
            for i in sorted(subject_ids))
        keypose_times = {}
        if len(keypose_frames) > 1 and shot_frames and shot_frames > 1:
            keypose_times = {label: _timecode(i * (shot_frames - 1) / (len(keypose_frames) - 1)) for i, label in enumerate(keypose_frames)}
        keypose_retention = "\n".join(
            f"{label} (key frame at {keypose_times[label]}): weak_reference - reproduced at that moment of the take; it is never shown as a still image, and the motion between key frames is continuous."
            if label in keypose_times else
            f"{label} (key frame in order): weak_reference - reproduced in that order within one continuous take; it is never shown as a still image, and the motion between key frames is continuous."
            if label in keypose_frames else
            f"{label} (key poses, their order, body positions, camera height): weak_reference - the key poses, their order, the stand-ins' positions, left-right order, occlusion and camera height are followed; smooth motion is generated between the poses; the stand-ins' surfaces and colors are replaced by the referenced characters and never appear; no extra people appear beyond those referenced."
            for label in keypose_labels)
        retention = subject_retention
        if keypose_retention:
            retention = (retention + "\n" if retention else "") + keypose_retention
        detail = text
        if keypose_times:
            detail = text.rstrip() + " The key frames are consecutive moments of this single continuous take, at: " + "; ".join(f"{t} {label}" for label, t in keypose_times.items()) + "."
        compiled = ("subject_definitions:\n" + "\n".join(definitions) + "\n\nsummary:\n" + text
                    + "\n\nretention_analysis:\n" + (retention or "Preserve the explicitly referenced visual content.")
                    + "\n\ndetailed_description:\n" + detail
                    + "\n\noverall_soundscape:\n" + soundscape + "\n\nnon_diegetic_music:\n" + music)
    else:
        compiled = "subject_definitions:\n" + "\n".join(definitions) + "\n\ndescription:\n" + text
    return compiled, blocks, items, "\n".join(report)


def attach_refs(conditioning, blocks, report, compiled):
    result = []
    for tensor, meta in conditioning:
        d = dict(meta)
        d["minimax_refs"] = list(blocks)
        d["bexplains_refpacks"] = {"version": VERSION, "reference_map": report, "prompt": compiled}
        result.append([tensor, d])
    return result
