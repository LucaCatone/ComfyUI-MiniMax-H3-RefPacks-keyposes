"""H3 reference preparation, storage, assembly and native prompt conditioning."""
import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

import folder_paths

from .core import (
    FORMAT,
    VERSION,
    add_pack,
    assemble,
    attach_refs,
    model_target,
    validate_latent,
    validate_audio_latent,
    validate_pack,
    reference_window,
    reference_layout,
    subject_numbers,
)
from .duration import h3_duration_frames
from .batching import enable_image_batches
from bexplains_refpacks.storage import load_pack, save_pack
from bexplains_refpacks.media import extract_media
from .diagnostics import log, reference_stats, profile_model
from bexplains_refpacks.nodes import (
    ROOT,
    _path,
    _files,
    BExplainsSaveRefPack,
    BExplainsLoadRefPack,
    BExplainsAddRefPack,
    BExplainsRefPacksPreviewText,
)
from .legacy import NODE_CLASS_MAPPINGS as LEGACY_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as LEGACY_NAMES

CATEGORY = "BExplains/RefPacks/H3"
PACK_TYPE = "BEXPLAINS_REFPACK"
SET_TYPE = "BEXPLAINS_REFERENCES"
def _target(value):
    return {"H3 Video": "h3_video", "H3 Image": "h3_image"}[value]


def _vae_target(vae, legacy_target=None):
    names = {cls.__name__ for cls in type(getattr(vae, "first_stage_model", None)).__mro__}
    if "PH3ImageVAE" in names:
        raise ValueError("p-h3-image is not supported. Connect a native MiniMax H3 Image or Video VAE.")
    if "MiniMaxH3ImageVAE" in names:
        return "h3_image"
    if "MiniMaxH3VideoVAE" in names:
        return "h3_image" if getattr(vae.first_stage_model, "still_frame", None) is not None or getattr(vae, "not_video", False) else "h3_video"
    # Retain old API calls for wrappers that previously supplied their target.
    if legacy_target is not None:
        return _target(legacy_target)
    raise ValueError("Cannot identify this VAE as MiniMax H3. Connect the native H3 Image or Video VAE.")


def _pack(name, target, entries, vae=None):
    encoder = getattr(vae, "first_stage_model", None)
    shared_image = all(e["kind"] == "image" for e in entries)
    return validate_pack({"format": FORMAT, "version": VERSION, "name": name, "target": "h3" if shared_image else target,
                          "source_target": target,
                          "encoder_class": (type(encoder).__module__ + "." + type(encoder).__name__) if encoder is not None else "unspecified",
                          "entries": entries})


def _resize_area(frames, megapixels, short_edge=None):
    from comfy_extras.nodes_minimax_h3 import _resize
    h, w = frames.shape[1:3]
    scale = min(1.0, math.sqrt(megapixels * 1_000_000 / (w * h)))
    if short_edge is not None:
        scale = min(scale, short_edge / min(w, h))
    tw, th = [max(32, round(side * scale / 32) * 32) for side in (w, h)]
    return _resize(frames[..., :3], tw, th, "disabled")


def _entry(kind, latent, vision, timestamps=None, **details):
    return {"kind": kind, "latent": validate_latent(latent, kind).detach().cpu().contiguous(),
            "vision": vision[..., :3].detach().cpu().float().contiguous(),
            **({"timestamps": list(timestamps)} if timestamps is not None else {}), **details}


class BExplainsH3EncodeImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "images": ("IMAGE",), "name": ("STRING", {"default": "character"}),
                             "reference_megapixels": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 8.0, "step": 0.05, "tooltip": "Max area for the encoded image; downscales only. Keep 0.3+ for identity (character sheets); 0.1 is enough for key poses."})}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Encode each image once and reuse the same saved pack with H3 Image or H3 Ref2VA. Preparation is detected from the connected native H3 VAE; no model selection is needed."

    def encode(self, vae, images, name, target=None, reference_megapixels=0.3):
        task = _vae_target(vae, target)
        entries = []
        if task == "h3_image":
            from comfy_extras.nodes_minimax_h3 import _image_refs, _size_short_edge
            # Use the native image model's separate vision and VAE sizing.
            refs = {f"image_{i+1}": images[i:i+1] for i in range(images.shape[0])}
            _, visions, blocks = _image_refs(refs, vae, lambda w,h: _size_short_edge(w,h,1024,1024), lambda w,h: _size_short_edge(w,h,1024,2048))
            entries = [_entry("image", b["latent"], v) for b,v in zip(blocks, visions)]
        else:
            for i in range(images.shape[0]):
                pixels = _resize_area(images[i:i+1], reference_megapixels, short_edge=2048)
                entries.append(_entry("image", vae.encode(pixels), pixels))
        return (_pack(name, task, entries, vae),)


class BExplainsH3EncodeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",), "frames": ("IMAGE",), "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "tooltip": "Source frame rate of the decoded frames; used only when duration is 0. Lowering it stretches the clip (5 frames at 1 fps -> ~124 frames). Prefer duration for discrete poses."}),
                             "name": ("STRING", {"default": "dance"}),
                             "reference_megapixels": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.05, "tooltip": "Max area for the encoded reference; downscales only, never upscales. 0.3 keeps ~416x736; 0.1 is enough for pose/motion references and is cheaper."}),
                             "frame_policy": (["pad final frame", "trim tail"], {"tooltip": "How to snap to the 17n+5 grid when the count is off: pad duplicates the last frame (keeps the ending), trim drops the tail (can lose up to 16 frames)."})},
                "optional": {"standin_map": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Which stand-in is which subject, e.g. 'the blue stand-in is <Character 1>; the red limbs are the camera body'. Stored in the pack and appended to the prompt definition."}),
                    "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 170.0, "step": 0.05,
                    "tooltip": "Target reference length in seconds. When > 0 it overrides fps and stretches or shrinks the input frames to the H3-aligned frame count (e.g. 5 s -> 124 frames)."})}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Resample video to H3's 24 fps, align to 17n+5 frames, encode once and retain the native 2 fps text-encoder vision frames. Audio is not included."

    def encode(self, vae, frames, fps, name, reference_megapixels, frame_policy, duration=0.0, standin_map=""):
        if frames.shape[0] < 2:
            raise ValueError("Provide decoded video frames, not a single still image.")
        if duration > 0:
            from .duration import h3_duration_frames
            n = h3_duration_frames(duration)
            indices = (torch.arange(n, device=frames.device) * frames.shape[0] / n).floor().long().clamp(max=frames.shape[0] - 1)
        else:
            n = max(5, round(frames.shape[0] * 24.0 / fps))
            indices = (torch.arange(n, device=frames.device) * fps / 24.0).floor().long().clamp(max=frames.shape[0]-1)
        frames = frames[indices]
        original = n
        if frame_policy == "pad final frame":
            while n % 17 != 5:
                n += 1
            if n > frames.shape[0]:
                frames = torch.cat([frames, frames[-1:].expand(n-frames.shape[0], -1, -1, -1)], dim=0)
        else:
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
        pixels = _resize_area(frames, reference_megapixels)
        indices = list(range(0, n, 12))
        entry = _entry("video", vae.encode(pixels), pixels[indices], [i/24.0 for i in indices],
                       frame_count=n, fps=24.0, original_resampled_frames=original, frame_policy=frame_policy,
                       standin_map=standin_map.strip())
        return (_pack(name, "h3_video", [entry], vae),)


class BExplainsH3EncodeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"audio_vae":("VAE",), "audio":("AUDIO",),
                            "name":("STRING",{"default":"audio"})}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Encode an audio reference with the H3 audio VAE using the native Ref2VA path. Chain directly for temporary audio, or save once as a reusable RefPack. Reference guidance is not a waveform passthrough."

    def encode(self, audio_vae, audio, name):
        waveform = audio.get("waveform")
        sr = audio.get("sample_rate")
        if not isinstance(waveform,torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1 or min(waveform.shape[1:]) < 1:
            raise ValueError("Provide one audio clip as waveform [1,channels,samples].")
        if not waveform.is_floating_point() or not torch.isfinite(waveform).all() or not isinstance(sr,(int,float)) or sr <= 0 or not math.isfinite(sr):
            raise ValueError("Audio needs a positive sample rate and finite floating-point samples.")
        from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
        z, length = _encode_ref_audio(audio_vae,audio)
        z = validate_audio_latent(z).detach().cpu().contiguous()
        entry = {"kind":"audio", "audio_latent":z, "ref_audio_t":int(length),
                 "duration_seconds":waveform.shape[-1]/sr, "source_sample_rate":sr,
                 "vae_sample_rate":getattr(audio_vae,"audio_sample_rate",32000)}
        return (_pack(name,"h3_video",[entry],audio_vae),)


class BExplainsH3PackLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "vision_images": ("IMAGE",),
                             "kind": (["image", "video"],),
                             "name": ("STRING", {"default": "reference"}), "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0})}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "pack"
    CATEGORY = CATEGORY
    DESCRIPTION = "Wrap an existing VAE encode with its matching reference pixels, without encoding again. Video frames must match the encoded 24 fps, 17n+5 frame input."

    def pack(self, latent, vision_images, kind, target=None, name="reference", fps=24.0):
        task = _target(target) if target is not None else ("h3" if kind == "image" else "h3_video")
        z = validate_latent(latent["samples"], kind)
        if kind == "image":
            if vision_images.shape[0] != 1:
                raise ValueError("Pack one image latent at a time. Use Encode H3 Image Refpack for multiple pictures.")
            entries = [_entry(kind,z,vision_images)]
        else:
            n = vision_images.shape[0]
            expected_t = 2 if n <= 5 else ((n-5)//17)*5+2
            if fps != 24 or n < 5 or n % 17 != 5 or z.shape[2] != expected_t:
                raise ValueError("Video pixels and latent must describe the same H3-aligned 24 fps clip. Use Encode H3 Video Refpack to prepare it automatically.")
            ix = list(range(0,n,12))
            entries = [_entry(kind,z,vision_images[ix], [i/24 for i in ix], frame_count=n, fps=24.0)]
        return (_pack(name,task,entries),)






class BExplainsH3ImportRefMod:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filename": ("STRING", {"default": "legacy.safetensors"}), "vae": ("VAE",),
                             "name": ("STRING", {"default": "imported"})},
                "optional": {"original_images_or_frames": ("IMAGE",)}}
    RETURN_TYPES = (PACK_TYPE,)
    RETURN_NAMES = ("refpack",)
    FUNCTION = "convert"
    CATEGORY = CATEGORY
    DESCRIPTION = "One-time import of a standalone full-encode H3 RefMod from models/refpacks. Original pixels are preferred; otherwise reconstruct them once using the matching VAE. Save the converted RefPack. Pooled/trained mods and bundles are rejected."

    def convert(self, filename, vae, target=None, name="imported", original_images_or_frames=None):
        with safe_open(str(_path(filename)), framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get("refmod_meta")
            if raw is None:
                raise ValueError("Expected a standalone H3 RefMod with refmod_meta. Use Pack Existing Latent for raw latent files.")
            info = json.loads(raw)
            if info.get("mode") not in ("encode", "full") or info.get("kind") not in ("image", "video"):
                raise ValueError("Only standalone full-encode image/video RefMods are supported. Recreate pooled or bundle references from original media.")
            z = f.get_tensor("latent").clone()
        kind = info["kind"]
        validate_latent(z,kind)
        pixels = original_images_or_frames
        reconstructed = pixels is None
        if reconstructed:
            pixels = vae.decode(z)
            if pixels.ndim == 5:
                pixels = pixels.reshape(-1, *pixels.shape[-3:])
        if kind == "image":
            pixels = pixels[:1]
        result = BExplainsH3PackLatent().pack({"samples":z}, pixels, kind, target, name, 24.0)[0]
        # The supplied VAE is used only for reconstruction, not latent creation.
        result["refmod_metadata"] = info
        result["reconstruction_vae_class"] = type(getattr(vae,"first_stage_model",None)).__module__ + "." + type(getattr(vae,"first_stage_model",None)).__name__ if reconstructed else "not used"
        result["vision_source"] = "VAE reconstruction (approximate)" if reconstructed else "original pixels"
        return (result,)




class BExplainsH3Duration:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seconds": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 170.0, "step": 0.01,
            "tooltip": "Requested video duration at H3's native 24 fps. Rounds up to 17n+5 frames (minimum 5). Connect length to the conditioning node's length input. 170 seconds stays within its 4096-frame limit."})}}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "calculate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Convert video duration in seconds to H3 frame count at 24 fps, rounded up to 17n+5. For example, 3 seconds gives 73 frames; 5 seconds gives 124 frames."

    def calculate(self, seconds):
        return (h3_duration_frames(seconds),)


class BExplainsH3RefPackConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "references": (SET_TYPE,),
                             "prompt": ("STRING", {"multiline": True, "default": "<Subject 1> performs the moves in <Video 1>"}),
                             "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                             "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                             "height": ("INT", {"default": 448, "min": 64, "max": 8192, "step": 8}),
                             "length": ("INT", {"default": 175, "min": 1, "max": 4096,
                                                  "tooltip": "Auto: 1-4 means independent images; 5+ means video frames. Video rounds up to 17n+5 at 24 fps, with a minimum of 5 frames. Image mode uses 1-4 images; larger lengths produce one image."}),
                             "auto_outfit": ("BOOLEAN", {"default": True})},
                "optional": {"vae": ("VAE",), "log_performance": ("BOOLEAN", {"default": True,
                    "tooltip": "Log reference sizes, text/vision encoding time and sampling time to the ComfyUI console. Does not alter reference tensors or sampling parameters."}),
                    "compress_motion": ("BOOLEAN", {"default": False,
                        "tooltip": "Off: use the leading reference window, capped at output length, like native H3. On: present the complete reference even for a shorter output; H3 may condense the routine. This is not deterministic retiming."}),
                    "allow_raw_prompt": ("BOOLEAN", {"default": False,
                        "tooltip": "Allow structured H3 section prompts or JSON with H3 section keys to pass through exactly as entered. Skips prompt expansion, aliases, auto-outfit and prompt tag checks. Plain shot instructions still compile normally; reference tensors remain attached."}),
                    "video_vision": ("BOOLEAN", {"default": True,
                        "tooltip": "On (default): send the reference video's sampled frames to the text encoder, which is what binds the motion to the subject. Off: keep one frame only; cheaper, but the subject often stays still because the latent alone does not carry the motion semantics. Shrink reference_megapixels instead if you need to save time."}),
                    "generation_mode": (["Auto", "Image", "Video"], {"default": "Auto",
                        "tooltip": "Auto: length 1-4 generates that many independent images; length 5+ generates video. Video mode requires an H3 video model and video-capable decoder; length is rounded up to 17n+5 at 24 fps, with 5 frames minimum (even when length is 1-4). Image mode forces still output: length 1-4 is the image batch count; larger lengths produce one image. The VAE no longer selects the mode."})}}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "reference_map_and_prompt")
    FUNCTION = "compile"
    CATEGORY = CATEGORY
    DESCRIPTION = "Assemble all references before fresh prompt encoding. Present cached images/video frames to H3's text encoder and attach the matching cached VAE latents. No VAE runs here."

    def compile(self, model, clip, references, prompt, negative_prompt, width, height, length, auto_outfit, log_performance=True, compress_motion=False, vae=None, allow_raw_prompt=False, generation_mode="Auto", video_vision=True):
        started = time.perf_counter()
        task = model_target(model)
        fallback = ""
        if not isinstance(length, int) or length < 1:
            raise ValueError("Length must be a positive integer.")
        if generation_mode not in ("Auto", "Image", "Video"):
            raise ValueError("Generation mode must be Auto, Image or Video.")
        intent = "h3_image" if generation_mode == "Image" or (generation_mode == "Auto" and length <= 4) else "h3_video"
        if vae is not None:
            decoder_intent = _vae_target(vae)
            if intent == "h3_video" and decoder_intent == "h3_image":
                raise ValueError("Video output requires an H3 Video VAE. Connect a video decoder or select Image mode.")
        if width < 64 or height < 64 or width % 16 or height % 16:
            raise ValueError("H3 output width and height must be multiples of 16 (at least 64).")
        for group in references.get("groups", []):
            family = group.get("target", group.get("source_target", "h3"))
            if family not in ("h3", "h3_image", "h3_video"):
                raise ValueError(f"Pack '{group['name']}' is for {family}. Re-encode it using an H3 VAE.")
        if intent == "h3_video":
            if task == "h3_image":
                raise ValueError("The H3 Image model cannot generate video. Select Image mode (or Auto with length 1-4), or connect an H3 video model.")
            from comfy_extras.nodes_minimax_h3 import _empty_av_latent
            latent, actual_length = _empty_av_latent(width,height,length)
            references = reference_window(references,actual_length,compress_motion)
            batch_size = 1
        else:
            batch_size = length if length <= 4 else 1
            actual_length = 1
            if length > 4:
                fallback = f"Image mode selected: length {length} exceeds the 4-image limit; generating one image."
                log(fallback)
            if task == "h3_video":
                # The video architecture still requires its audio stream. Each batch item
                # has a single visual frame, never multiple time steps for multiple images.
                from comfy_extras.nodes_minimax_h3 import _av_latent
                latent = _av_latent(width, height, 1, 2, batch_size)
            else:
                import comfy.model_management
                latent = {"samples": torch.zeros([batch_size,24,1,height//16,width//16],device=comfy.model_management.intermediate_device())}
        compiled, blocks, items, mapping = assemble(references,prompt,task,auto_outfit,allow_raw_prompt=allow_raw_prompt,shot_frames=actual_length if intent == "h3_video" else None, video_vision=video_vision)
        tokenize_args = {"minimax_ref_items": items} if task == "h3_video" else {"images": [i["data"] for i in items], "picture_label": "Picture %d: "}
        if intent == "h3_video":
            mapping += "\nReference policy: " + ("Compress Motion — complete source clips" if compress_motion else "native leading window, capped at output length")
            for group in references["groups"]:
                for e in group["entries"]:
                    if e["kind"] == "video":
                        mapping += f"\n{group['name']}: {e.get('source_frame_count',e.get('frame_count','unknown'))} source frames -> {e.get('frame_count','unknown')} reference frames used"
        if intent == "h3_image" and task == "h3_video":
            mapping += "\nReference policy: single-frame output; complete input references retained."
        assembled = time.perf_counter()
        if log_performance:
            log(f"Reference assembly and output latent allocation: {assembled-started:.3f}s")
            log("Positive text/vision tokenization starting.")
        tokens = clip.tokenize(compiled, **tokenize_args)
        tokenized = time.perf_counter()
        if log_performance:
            log(f"Positive tokenization: {tokenized-assembled:.3f}s; positive text/vision encoding starting.")
        positive = clip.encode_from_tokens_scheduled(tokens)
        encoded = time.perf_counter()
        if log_performance:
            log(f"Positive text/vision encoding: {encoded-tokenized:.3f}s; negative encoding starting.")
        if not positive or any("minimax_token_tags" not in c[1] for c in positive):
            raise ValueError("CLIP did not produce native H3 token tags. Load the H3 Qwen3-VL encoder with CLIPLoader type minimax.")
        positive = attach_refs(positive,blocks,mapping,compiled)
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        negative_encoded = time.perf_counter()
        duration = f" ({actual_length / 24.0:.3f} seconds at 24 fps)" if intent == "h3_video" else ""
        details = mapping + (
            f"\n\nGeneration mode: {generation_mode}\n"
            f"conditioning architecture: {task}\n"
            f"output intent: {intent}\n"
            f"batch size: {batch_size}\n"
            f"output frames: {actual_length}{duration}\n"
            f"canvas: {width//16*16}x{height//16*16}\n\n"
        ) + compiled
        if fallback:
            details += "\n" + fallback
        if log_performance:
            stats = reference_stats(blocks,items,latent,positive[0][0].shape[1])
            for line in stats:
                log(line)
            log(f"Negative tokenization + encoding and metadata attachment: {negative_encoded-encoded:.3f}s")
            log(f"Conditioning node total before profiler setup: {negative_encoded-started:.3f}s")
            model = profile_model(model)
            details += "\n\nPERFORMANCE\n" + "\n".join(stats) + f"\nPositive encoding: {encoded-tokenized:.3f}s; conditioning total: {negative_encoded-started:.3f}s\nSampling timings will appear in the ComfyUI console."
        if intent == "h3_image" and batch_size > 1:
            model = enable_image_batches(model)
            details += "\nH3 image batch: native batch-one forward passes are collected per denoising step."
        return (model,positive,negative,latent,details)


class BExplainsRefPackToH3(BExplainsH3RefPackConditioning):
    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"]["vae"] = schema["optional"].pop("vae")
        return schema
    RETURN_TYPES = BExplainsH3RefPackConditioning.RETURN_TYPES + ("VAE",)
    RETURN_NAMES = BExplainsH3RefPackConditioning.RETURN_NAMES + ("vae",)
    DESCRIPTION = "H3 receiver: Auto selects images for length 1-4 and video for length 5+. Image/Video explicitly overrides Auto. Image lengths 1-4 are independent batch items with T=1; greater lengths fall back to one image. The model still selects native conditioning format. Connect the VAE output to VAEDecode."

    def compile(self, model, clip, references, prompt, negative_prompt, width, height, length, auto_outfit, vae, log_performance=True, compress_motion=False, allow_raw_prompt=False, generation_mode="Auto", video_vision=True):
        result = super().compile(model, clip, references, prompt, negative_prompt, width, height, length, auto_outfit,
                                 log_performance=log_performance, compress_motion=compress_motion, vae=vae, allow_raw_prompt=allow_raw_prompt, generation_mode=generation_mode, video_vision=video_vision)
        return result + (vae,)


class BExplainsInspectRefPacks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"references":(SET_TYPE,)},
                "optional":{"prompt":("STRING",{"default":"", "multiline":True,
                    "tooltip":"Optional shot instruction. Leave empty to inspect only the tag map, even before writing a valid prompt."}),
                    "prompt_format":(["H3 Video","H3 Image"],{"tooltip":"Only controls the optional compiled prompt preview; no model or text encoder is loaded."}),
                    "auto_outfit":("BOOLEAN",{"default":True}),
                    "allow_raw_prompt":("BOOLEAN",{"default":False, "tooltip":"Match Allow Raw Prompt on the H3 encoder to preview the exact structured prompt without expansion."}),
                    "length":("INT",{"default":0, "min":0, "max":4096, "tooltip":"Output frame count. Set it to the encoder's length so key frame timecodes in the preview match the generated clip."}),
                    "vae":("VAE",{"tooltip":"Matching image/video VAE to reconstruct full visual latents; omit to return cached reference pixels."}),
                    "audio_vae":("VAE",{"tooltip":"Matching H3 audio VAE for reconstructing stored audio."})}}
    RETURN_TYPES = (SET_TYPE,"STRING","STRING","IMAGE","AUDIO")
    RETURN_NAMES = ("references","tag_mapping","compiled_prompt","Images","Audio")
    OUTPUT_IS_LIST = (False,False,False,True,True)
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Inspect every Picture/Video/Audio slot and its source pack before text encoding. Run only this node to avoid downstream encoding/sampling. Optional prompt preview uses the same compiler; match prompt format and auto_outfit to the conditioning node."

    def inspect(self,references,prompt="",prompt_format="H3 Video",auto_outfit=True,allow_raw_prompt=False,audio_vae=None,vae=None,length=0):
        layout=list(reference_layout(references))
        lines=["REFERENCE TAG MAP — native presentation order", ""]
        for gi,ei,e,label in layout:
            group=references['groups'][gi]
            origin=group.get('source_file') or '(temporary / unsaved)'
            lines.append(f"{label} -> {origin} | pack: {group['name']} | entry {ei+1} of {len(group['entries'])} | {group['role']} {group['role_id']}")
        lines += ["", "ROLE MAPPINGS"]
        subjects = subject_numbers(references)
        for gi,g in enumerate(references['groups']):
            tags=[label for index,_,_,label in layout if index == gi]
            alias={"subject":"Subject","character":"Character","item":"Item","motion":"Motion","keyposes":"Poses","outfit":"Outfit","location":"Location","style":"Style"}.get(g['role'])
            title=f"<{alias} {g['role_id']}>" if alias else f"{g['role']} {g['role_id']}"
            subject = f"<Subject {subjects[gi]}> -> " if g["role"] in ("character", "item") else ""
            lines.append(f"{title} -> {subject}{', '.join(tags)}")
        lines += ["", "PACK COUNTS"]
        for g in references['groups']:
            counts={kind:sum(e['kind']==kind for e in g['entries']) for kind in ('image','video','audio')}
            lines.append(f"{g.get('source_file') or g['name']}: {counts['image']} picture(s), {counts['video']} video(s), {counts['audio']} audio clip(s)")
        mapping='\n'.join(lines)
        compiled=""
        preview_error=""
        if prompt.strip():
            try:
                compiled=assemble(references,prompt,_target(prompt_format),auto_outfit,allow_raw_prompt=allow_raw_prompt,shot_frames=length or None)[0]
            except ValueError as error:
                preview_error=f"\n\nPROMPT PREVIEW ERROR: {error}\nThe tag map above remains available; correct the prompt and inspect again."
        display=mapping + (preview_error or (f"\n\nCOMPILED PROMPT PREVIEW ({prompt_format})\n{compiled}" if compiled else "\n\nAdd an optional prompt to preview its compiled form. No text encoder has run."))
        images,audio,media_report=extract_media(references,audio_vae,vae)
        display += "\n\n" + media_report
        return {"ui":{"text":[display]},"result":(references,mapping,compiled,images,audio)}




CLASSES = [BExplainsRefPackToH3,BExplainsH3Duration,BExplainsH3EncodeImages,BExplainsH3EncodeVideo,BExplainsH3EncodeAudio,BExplainsH3PackLatent,
           BExplainsH3ImportRefMod,BExplainsH3RefPackConditioning,BExplainsInspectRefPacks]
NODE_CLASS_MAPPINGS = {**LEGACY_CLASSES, **{c.__name__:c for c in CLASSES}}
NODE_DISPLAY_NAME_MAPPINGS = {**LEGACY_NAMES,
    "BExplainsRefPackToH3": "BExplains RefPack to H3 Encoder",
    "BExplainsH3Duration": "BExplains H3 Duration",
    "BExplainsH3EncodeImages": "Encode H3 Image Refpack",
    "BExplainsH3EncodeVideo": "Encode H3 Video Refpack",
    "BExplainsH3EncodeAudio": "Encode H3 Audio Refpack",
    "BExplainsH3PackLatent": "BExplains Pack Existing H3 Latent",
    "BExplainsSaveRefPack": "BExplains Save RefPack",
    "BExplainsLoadRefPack": "BExplains Load RefPack",
    "BExplainsH3ImportRefMod": "BExplains Import H3 RefMod",
    "BExplainsAddRefPack": "BExplains Add RefPack",
    "BExplainsH3RefPackConditioning": "BExplains H3 RefPack Conditioning",
    "BExplainsRefPacksPreviewText": "BExplains RefPacks Preview Text",
    "BExplainsInspectRefPacks": "BExplains Inspect Refpacks",
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v for k, v in NODE_DISPLAY_NAME_MAPPINGS.items() if k in NODE_CLASS_MAPPINGS}
