# MiniMax H3 RefPacks — keyposes fork

A fork of [BExplains MiniMax H3 RefPacks](https://github.com/BExplains/ComfyUI-MiniMax-H3-RefPacks)
that adds a **`keyposes`** role plus a few quality-of-life tweaks.

**`keyposes`** accepts either still key frames (image pack) or a single pose video. The compiled prompt
treats them as discrete key poses held at their moment — one continuous take, motion generated between
poses — never as portraits or separate shots. Each frame is labelled with its timestamp in
`retention_analysis`, and `detailed_description` gets a compact timing list spread across the shot
length, so the reference no longer has to match the output duration.

Other changes over upstream:

- `Encode H3 Video Refpack` gains a **`duration`** (seconds) input: stretch or shrink the pose video to
  the H3-aligned frame count without fps math (`5 s` → `124` frames, `10 s` → `243`).
- The encoder has a **`video_vision`** toggle: when off, the reference video is kept out of the text
  encoder (one frame only). Cheaper, but the subject often freezes — the latent alone does not carry the
  motion semantics. Shrink `reference_megapixels` instead when you need to save time.
- The video encoder also takes a **`proxy_colors`** field: say which coloured proxy is which subject
  (`the blue mannequin is <Character 1>; the red limbs are the camera body`). It is stored in the pack and
  appended to the prompt definition, aliases resolved.
- Tooltips on `fps`, `duration`, `frame_policy`, `proxy_colors` and `reference_megapixels`.
- Installs from `git clone` as-is: the shared package is discovered under its `-keyposes` folder name.

---

## Ideal workflow for `keyposes`

End-to-end recipe for turning a handful of static poses into a moving shot. The numbers below are the
configuration that was tested; treat them as defaults, not laws.

### 1. Produce the pose frames

- **One still per key pose.** No animation needed: a batch of 5 stills is enough. The encoder stretches
  them over the whole shot by duplicating frames, so each pose is held for its share of the timeline.
- One **coloured proxy** per body. One colour per whole body, distinct colours between bodies
  (e.g. blue = the partner, pink = the camera body). Never reuse a colour for two bodies.
- **Neutral background**: no room, no furniture detail, no textures. The room comes from the character
  and location references, not from the pose frames.
- Because the encoder duplicates frames, the reference contains hard cuts between holds rather than
  real motion. That is expected: the `keyposes` wording exists to tell the model to generate the motion
  between the poses. Adding 2-4 frame crossfades before encoding is optional, not required.
- Resolution: **0.2-0.3 MP per frame** is enough. Below that the pose stops being legible and the
  subject freezes instead of moving.
- Optional: the exact opening frame, reused later as an `AddGuide` on frame 0.

### 2. Encode the pack

| Field | Value | Why |
|---|---|---|
| `duration` | **the same length as the encoder's `length`** | stretches the poses over the whole shot; if it is shorter, the tail of the shot has no reference and the last pose is dropped |
| `proxy_colors` | `the blue mannequin is <Character 1>; the pink limbs are the camera body` | binds each proxy to its subject |
| `reference_megapixels` | 0.2-0.3 | 0.1 is too coarse for body poses; 0.3 is the tested size |
| `frame_policy` | pad final frame | keeps the last pose instead of trimming it |

### 3. Wire it

`Encode H3 Video Refpack` -> `Add RefPack` with role **keyposes** -> `RefPack to H3 Encoder`.

`motion` is for continuous footage, `keyposes` is for discrete poses. Mixing them up gives either a
frozen subject or a stuttering one.

### 4. Write the instruction

Say **what happens, in order**, and name the subjects. Leave the timing to the compiler.

```text
A single continuous first-person shot: a young man sits on the sofa watching the television, then turns
his head to the left toward <Character 1>, who leans in and rests a hand on the backrest. <Poses 1>
defines the poses and their order.
```

The compiler adds the pose definitions, one retention line per pose, and the timing list.

### 5. Troubleshooting

| Symptom | Likely cause |
|---|---|
| The subject stays perfectly still | `reference_megapixels` too low (0.1 destroys the pose), or `proxy_colors` missing so nothing binds the motion to the character |
| A proxy colour becomes a prop (blue slipper, cyan patch) | `proxy_colors` missing, or a saturated colour at the frame edge; add the mapping and crop the edge |
| The last pose is never reached | the reference is shorter than the output: check `frame_count` under `PACK COUNTS` in the Inspector against your `length`, and set `duration` to match; if it still misses, add one `AddGuide` on the last frame |
| The motion stutters, bodies duplicate | the reference was decimated (frames skipped) instead of held; re-encode with holds and transitions |
| Room or sofa comes from the reference | the pose frames carry a background; use a neutral background or crop |
| Everything looks grey or plastic | no style sentence in the instruction; add one ("realistic handheld footage, natural daylight") |

### 6. Checklist

- [ ] one distinct proxy colour per body, neutral background
- [ ] `duration` equals the output length (compare `frame_count` in the Inspector with `length`)
- [ ] `proxy_colors` filled and visible under `PACK COUNTS` in the Inspector
- [ ] character pack at 0.3 MP or more, pose pack at 0.2-0.3 MP
- [ ] instruction names subjects and order; no timestamps needed
- [ ] `video_vision` on

---

**Reusable image, video and audio references for MiniMax H3, with role-aware prompting and complete scene reuse.**

Prepare your characters, objects, outfits, locations, motion clips and audio once. Assign their roles, describe the shot, and let the H3 receiver bind the cached references to native H3 conditioning.

The collection builds on **[BExplains RefPacks](https://github.com/BExplains/ComfyUI-RefPacks)**, which supplies individual pack loading/saving, reference grouping and self-contained scene files. Install both packages alongside each other.

**Version 1.0.1.** The shared package's Flux 2 and LTXV adapters remain experimental testing integrations. This collection handles H3-specific preparation, prompt formats, temporal rules and output intent.

## Contents

- [Features](#features)
- [Requirements and installation](#requirements-and-installation)
- [Quick start](#quick-start)
- [Node reference](#node-reference)
- [Prompting](#prompting)
- [How it works](#how-it-works)
- [Compatibility](#compatibility)
- [Resolution and performance](#resolution-and-performance)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Reporting problems](#reporting-problems)
- [Credits](#credits)

## Features

- Encode H3 still images, reference videos and standalone reference audio once.
- Reuse prepared still-image packs with the H3 Image and H3 video model paths.
- Send cached visual reference pixels to H3's text encoder and cached VAE data to diffusion conditioning.
- Compile plain shot instructions into H3's image or video prompt format.
- Give Character and Item groups distinct subject definitions.
- Resolve `<Character n>`, `<Item n>`, `<Motion n>`, `<Outfit n>`, `<Location n>` and `<Style n>` aliases automatically.
- Inspect the exact Picture/Video/Audio slots and compiled prompt before encoding.
- Pass a structured text or JSON prompt through unchanged when **Allow Raw Prompt** is enabled.
- Choose Auto, Image or Video generation explicitly, while retaining model-specific conditioning.
- Generate 1–4 independent image batch items using a wrapper around native H3's batch-one forward passes.
- Convert seconds into valid H3 video frame counts with **H3 Duration**.
- Preserve existing H3 node IDs and existing version-1 RefPacks.

Reference encoding is not LoRA training, and role labels do not guarantee identity, outfit or motion fidelity for every seed.

## Requirements and installation

Requires current ComfyUI with native MiniMax H3 support and the shared **BExplains RefPacks** package. Use matching H3 model weights, H3 Qwen3-VL text encoder and VAEs for your workflow.

### Install from the Comfy Registry

Install with ComfyUI-Manager or Comfy CLI:

```bash
comfy node install bexplains-refpacks
comfy node install bexplains-h3-refpacks
```

[Registry listing](https://registry.comfy.org/publishers/bexplains/nodes/bexplains-h3-refpacks). New releases must complete registry security processing before an active release is available.

The H3 collection requires the shared **BExplains RefPacks** custom-node package. Install it separately; Python dependency installation does not install the companion node pack. Keep only one installed copy of each package, then restart ComfyUI.

### Install from source

From your ComfyUI installation directory:

```bash
cd custom_nodes
git clone https://github.com/BExplains/ComfyUI-RefPacks.git BExplains_RefPacks
git clone https://github.com/BExplains/ComfyUI-MiniMax-H3-RefPacks.git BExplains_H3_RefPacks
```

Skip either clone if that package is already installed. These commands retain the legacy destination names. The H3 collection also discovers the shared package under its registry name `bexplains-refpacks` or source-clone name `ComfyUI-RefPacks`, including configured custom-node directories.

If needed, run this from the ComfyUI directory using ComfyUI's Python environment:

```bash
python -m pip install -r custom_nodes/BExplains_RefPacks/requirements.txt
```

Restart ComfyUI and refresh the browser. H3 nodes are under **BExplains / RefPacks / H3**; shared save/load/add/preview nodes are under **BExplains / RefPacks**.

For archive installation, copy each package into its same-named folder under `custom_nodes`. Each folder must directly contain its `__init__.py`. When upgrading from the old combined package, replace the old shared code folder and install this H3 collection; leave saved assets in `models/refpacks/`.

### What is needed for each stage?

| Stage | Required inputs |
| --- | --- |
| Prepare image references | Source images and a native H3 Image or H3 Video VAE |
| Prepare video references | Decoded video frames, their source FPS and an H3 Video VAE |
| Prepare audio references | One audio clip and the H3 audio VAE |
| Compile and generate a shot | H3 model, H3 Qwen3-VL CLIP/text encoder, reference set, decoding VAE and normal sampling nodes |

Preparation does not require the diffusion model or Qwen encoder. Final generation does. Select `minimax` as the CLIP loader type for the H3 text encoder.

For image output with a video diffusion model, the receiver allocates the architecture's audio stream internally. You do not need an audio VAE just to decode the image. Audio decoding needs its own appropriate VAE when you actually use the generated audio.

## Quick start

### Prepare and save a character

1. Add **Encode H3 Image Refpack**.
2. Connect a native H3 VAE and your reference image or image batch.
3. Enter a name such as `mai_concert`.
4. Connect `refpack` to the shared **BExplains Save RefPack** node.
5. Use the same plain name for Save's filename and queue the save node.

This creates `ComfyUI/models/refpacks/mai_concert.safetensors`. The extension is automatic, and overwriting is off by default.

Each IMAGE batch member becomes a separate reference entry. A character sheet supplied as one image remains one reference containing several views; the node does not split it into crops.

### Assemble references and generate a shot

1. Load the saved asset with **Load RefPack**.
2. Connect it to **Add RefPack** and choose `character`.
3. Chain more Add nodes for an item, outfit, location, motion or audio as needed.
4. Connect the final `references` to **BExplains RefPack to H3 Encoder**.
5. Connect the H3 model, text encoder and the VAE intended to decode the output.
6. Enter a shot instruction, dimensions and integer length.
7. Connect model, positive, negative and latent outputs to your sampler.
8. Connect the sampler's output to VAEDecode and the receiver's `vae` output to the decoder's VAE input.

Use the sampler settings known to work with your checkpoint. Distilled/turbo examples do not establish suitable steps or CFG for a base checkpoint.

### Reuse the entire scene

Connect the final assembled references to **Save RefPack Bundles**, enter `concert_opening`, and queue the save. It creates:

```text
ComfyUI/models/refpacks/references/concert_opening.references.safetensors
```

For another shot, use **Load RefPack Bundles** and connect it directly to the H3 receiver or inspector. The file contains all pack data, roles and ordering; the original individual pack files are not required. It does not include prompts, samplers or model weights.

### Inspect before sampling

Connect the reference set to **BExplains Inspect Refpacks**. Its map identifies each `<Picture n>`, `<Video n>` and `<Audio n>` with the source pack and local entry number. Enter an optional instruction to preview its compiled prompt without loading or running the text encoder.

Match prompt format, Auto Outfit and Allow Raw Prompt to the receiver. Inspector controls do not modify the receiving node's inputs.

## Node reference

### Preparation nodes

| Node / internal ID | Main inputs | Output and behavior |
| --- | --- | --- |
| **Encode H3 Image Refpack** / `BExplainsH3EncodeImages` | `vae`, `images`, `name`, `reference_megapixels` | One RefPack with a separate entry for each batch image; VAE type selects preparation path |
| **Encode H3 Video Refpack** / `BExplainsH3EncodeVideo` | `vae`, decoded `frames`, source `fps`, `name`, `reference_megapixels`, `frame_policy` | One video RefPack; resamples to 24 fps and pads or trims to `17n + 5` |
| **Encode H3 Audio Refpack** / `BExplainsH3EncodeAudio` | `audio_vae`, `audio`, `name` | One audio RefPack; native audio encoding/resampling path |
| **BExplains Pack Existing H3 Latent** / `BExplainsH3PackLatent` | `latent`, matching `vision_images`, `kind`, `name`, `fps` | Wraps matching cached data without another VAE encode |
| **BExplains Import H3 RefMod** / `BExplainsH3ImportRefMod` | `filename`, `vae`, `name`, optional original images/frames | Limited one-time import of standalone full-encode image/video RefMods |

Preparation outputs `BEXPLAINS_REFPACK`. Use the shared Save/Load RefPack and Add RefPack nodes to store and assemble it.

Pack Existing Latent accepts one image reference at a time. For video, pixels and latent must describe the same 24 fps, H3-aligned clip. Import RefMod accepts standalone `encode`/`full` image/video data, not trained/pooled mods or bundle containers. Supply matching original pixels when available; otherwise import reconstructs approximate vision pixels once with the provided VAE. A file from the earlier image-RefMod bundle creator is not automatically a compatible standalone import.

### BExplains RefPack to H3 Encoder

Internal ID: `BExplainsRefPackToH3`.

| Input | Meaning |
| --- | --- |
| `model` | Determines native image/video conditioning and stream layout |
| `clip` | H3 Qwen3-VL text encoder, loaded with type `minimax` |
| `references` | Assembled or loaded reference set |
| `vae` | Intended output-decoding VAE; must support the selected output mode |
| `generation_mode` | Auto: length 1–4 means images; 5+ means video. Image/Video forces that output type |
| `prompt` | Shot instruction, or an allowed structured raw prompt |
| `negative_prompt` | Passed to the text encoder unchanged |
| `width`, `height` | Output canvas dimensions |
| `length` | Image count or video frame count, according to generation mode |
| `auto_outfit` | Add an outfit instruction when there is exactly one possible wearer and one outfit |
| `log_performance` | Log reference sizes and encoding/sampling timing information |
| `compress_motion` | Present complete video references instead of the default leading window |
| `allow_raw_prompt` | Allow recognized structured text/JSON to bypass prompt rewriting |

Outputs, in order: `model`, `positive`, `negative`, `latent`, `reference_map_and_prompt`, `vae`.

The output VAE is the same object supplied at the input. Connecting it to VAEDecode keeps output intent and decoding aligned. The receiver does not VAE-encode generated pixels.

#### Image versus video intent

Select **generation_mode** on the H3 encoder. Its tooltip explains the rules:

| Mode | Length | Output |
| --- | --- | --- |
| Auto | 1–4 | That many independent image batch items, each with temporal length 1 |
| Auto | 5+ | Video, rounded upward to the `17n + 5` frame grid at 24 fps |
| Image | 1–4 | That many independent image batch items |
| Image | Above 4 | One image, with an explanation in the report |
| Video | Any positive integer | Video, rounded upward to `17n + 5`, with a minimum of 5 frames |

The VAE no longer selects generation mode. Connect a matching decoder. Video output requires a video-capable H3 VAE and H3 video diffusion model; incompatible combinations are rejected before text encoding. The H3 video model can still generate images. The dedicated H3 Image model cannot generate video.

For older workflows that used an image VAE with a length above 4, explicitly choose **Image** to retain still output. For video workflows with lengths 1–4, explicitly choose **Video**. Existing workflows with image lengths 1–4 or video lengths 5+ can use Auto.

Native H3 diffusion forwards accept only batch size one. For multiple images, this collection processes each batch item separately during each denoising forward, then concatenates the predictions back into the sampler's batch. Images use separate batch items and noise; they are not encoded as consecutive temporal frames. This is not a claim of native parallel batch acceleration.

### BExplains H3 Duration

Internal ID: `BExplainsH3Duration`.

Input: FLOAT `seconds`. Output: INT `length` for the receiving encoder.

The calculator uses 24 fps and rounds upward to the first valid `17n + 5` count:

| Requested duration | Output length | Effective duration |
| --- | --- | --- |
| 3 seconds | 73 frames | About 3.042 seconds |
| 5 seconds | 124 frames | About 5.167 seconds |
| 10 seconds | 243 frames | About 10.125 seconds |

The UI accepts 0.01–170 seconds, staying within the receiver's 4096-frame input limit. It calculates video duration; it does not choose output intent or represent image batch count.

### BExplains Inspect Refpacks

Internal ID: `BExplainsInspectRefPacks`.

Inputs: `references`, optional `prompt`, `prompt_format`, `auto_outfit`, `allow_raw_prompt`, `vae`, `audio_vae`.

Outputs: original `references`, `tag_mapping`, `compiled_prompt`, `Images`, `Audio`. The report is also displayed directly on the node. Missing prompt tags are reported while keeping the reference map available. The inspector never runs Qwen or diffusion sampling. It only decodes latents when an optional VAE is connected.

### Older node IDs

**BExplains H3 RefPack Conditioning** (`BExplainsH3RefPackConditioning`) retains its original five outputs. It also supports Auto/Image/Video selection, with an optional decoding VAE for compatibility checks. Prefer RefPack to H3 Encoder for new workflows.

**BExplains Set H3 Reference Latent** (`BExplainsSetH3ReferenceLatent`) remains available for legacy workflows. It appends an H3 still-reference latent to existing conditioning; it does not add vision inputs to the text encoder or perform the new role-aware prompt compilation.

### Extract media from an inspected RefPack

Connect **Load RefPack → Add RefPack → Inspect**, or connect **Load RefPack Bundles** directly to an inspector. Both the shared metadata inspector and **BExplains Inspect Refpacks** expose `Images` and `Audio` after their original outputs.

- **Images without `vae`:** returns the stored reference pixels without decoding. H3 video references contain sampled vision frames, not every frame of the source clip.
- **Images with `vae`:** reconstructs image/video latents using the matching visual VAE. A video entry stays together as one IMAGE batch, suitable for video preview or re-encoding. Use the VAE matching the entries; mixed model-family sets may require inspecting separate sets.
- **Audio with `audio_vae`:** reconstructs each audio entry through ComfyUI's native audio decoder. The waveform and output sample rate are returned as standard AUDIO data. This is a VAE reconstruction, not the original waveform.
- **Audio without `audio_vae`:** returns an empty audio list and explains in the report that an audio VAE is needed. Metadata inspection and image extraction still work.

Both VAE inputs are optional. Leave them disconnected for metadata-only inspection or when their decoded output is not needed. `Images` and `Audio` are list outputs: each visual entry retains its dimensions and each audio clip stays separate. Connect them to Preview Image/Preview Audio or the appropriate Refpack encoder; downstream nodes run for each list item. An absent media type produces an empty list, not a fabricated black image or silent clip. Pixels and decoded audio can be re-encoded, but do not recover detail discarded during the original preparation or VAE encoding.

## Prompting

### Character and Item references

Assign the woman pack the role `character` and the microphone pack the role `item`, then write:

```text
<Character 1> holds <Item 1> in her hand.
```

The compiler creates a person-specific definition for the character and an object-specific definition for the item. With those assets added in that order, the instruction becomes:

```text
<Subject 1> holds <Subject 2> in her hand.
```

Character and Item alias counters are independent, while their native Subject IDs share a namespace with generic subjects. If you add the item first, its Subject ID comes first. Use the inspector rather than assuming a Picture or Subject number.

### Roles and aliases

| Add RefPack role | Instruction tag | Native interpretation |
| --- | --- | --- |
| `character` | `<Character n>` | A `<Subject m>` with person identity/appearance guidance |
| `item` | `<Item n>` | A `<Subject m>` with object shape, material, color and detail guidance |
| `subject` | `<Subject n>` | Generic subject definition, without assuming a person |
| `outfit` | `<Outfit n>` | The outfit group's actual Picture slots |
| `location` | `<Location n>` | The location group's actual Picture slots |
| `style` | `<Style n>` | The style group's actual Picture slots |
| `motion` | `<Motion n>` | The motion group's actual Video slots |
| `video` | `<Video n>` | Native video slot; numbered across all video entries |
| `audio` | `<Audio n>` | Native audio slot |
| `reference` | `<Picture n>` for images | General visual reference, without a prescribed subject definition |

Multiple views within one Character or Item group define one subject. Multiple videos within a Motion group expand to all of that group's Video slots. Native reference presentation orders images first, then videos, then audio.

For example:

```text
<Character 1> performs the movement shown in <Motion 1>, holding <Item 1> in her hand, wearing <Outfit 1>, in <Location 1>.
```

Motion is role-aware prompt guidance. It does not change the model's underlying motion-control capabilities or guarantee exact choreography transfer.

### Auto Outfit

With exactly one Character or generic Subject group, exactly one outfit group, and no explicit mention of that outfit, Auto Outfit adds a clothing instruction.

Item groups do not count as wearers. When an item is also present, the added sentence explicitly identifies the wearer's Subject ID. With multiple possible wearers, the compiler does not choose one automatically. Turn Auto Outfit off to keep the reference available without that automatic instruction.

### Compiled prompt formats

Plain instructions are expanded into the format selected by the diffusion model architecture:

| Model path | Generated sections |
| --- | --- |
| H3 Image | `subject_definitions`, `description` |
| H3 video architecture | `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music` |

Using the video model for a still image retains its video-architecture prompt format. Output intent and conditioning format are separate decisions.

### Allow Raw Prompt

Enable **Allow Raw Prompt** when you want to supply your own compiled/structured prompt. A recognized raw prompt is sent to the text encoder **exactly as entered**, including whitespace.

The node does not wrap it, strip it, serialize it again, expand aliases, append an outfit instruction or validate its prompt tags. It still assembles native reference inputs and attaches the cached tensor data. Model and tensor compatibility checks remain active. The negative prompt is unchanged, as before.

| Detected input | Toggle off | Toggle on |
| --- | --- | --- |
| Structured H3 text or recognized JSON | Error explaining how to enable raw mode | Exact passthrough |
| Ordinary shot instruction | Compiled normally | Compiled normally |

Detection is structural, not semantic. A text line beginning with a recognized section name followed by a colon is sufficient. Leading spaces and capitalization are ignored for text headers.

Recognized sections are `subject_definitions`, `description`, `summary`, `detailed_description`, `retention_analysis`, `overall_soundscape` and `non_diegetic_music`.

For example, this is a raw text prompt when the toggle is enabled:

```text
subject_definitions:
<Subject 1> is the woman shown in <Picture 1>.
<Subject 2> is the microphone shown in <Picture 2>.

description:
<Subject 1> holds <Subject 2> in her hand.
```

Raw mode does not expand `<Character n>`, `<Item n>`, `<Motion n>`, `<Outfit n>`, `<Location n>` or `<Style n>`. Use native Subject/Picture/Video/Audio tags and your own definitions. Verify that the referenced Picture/Video/Audio slots actually exist in the inspector's map.

### JSON prompts

A valid JSON object is recognized when it contains **at least one exact, lowercase top-level key** from the recognized sections above. No particular combination is required, and the node does not enforce a schema on the values or remove additional keys.

Image example:

```json
{
  "subject_definitions": "<Subject 1> is the woman shown in <Picture 1>. <Subject 2> is the microphone shown in <Picture 2>.",
  "description": "<Subject 1> holds <Subject 2> in her hand, standing in a sunny garden."
}
```

Video-architecture example:

```json
{
  "subject_definitions": "<Subject 1> is the woman shown in <Picture 1>. <Subject 2> is the microphone shown in <Picture 2>.",
  "summary": "<Subject 1> sings into <Subject 2> on a concert stage.",
  "detailed_description": "The camera slowly approaches as <Subject 1> raises <Subject 2> to her mouth.",
  "overall_soundscape": "Follow the requested vocal performance.",
  "non_diegetic_music": "No additional music is requested."
}
```

**JSON is passed as text, not converted into the usual H3 section format.** Successful detection does not guarantee that H3 interprets every JSON structure equally well. The examples use simple string values as a starting point for testing. Without a matching section key, an otherwise valid JSON object is treated as an ordinary instruction.

Enable the same raw toggle in Inspect References to preview exactly what the receiver will use. Inspector text is a preview; it does not automatically update the receiving prompt.

### Style references

Prepare and load an image RefPack, then select **style** on **Add RefPack**. In an H3 shot instruction, write:

```text
<Character 1> dances in the style of <Style 1>.
```

If the character occupies Picture 1 and the style pack occupies Picture 2, this becomes `<Subject 1> dances in the style of <Picture 2>.` A style pack with several images expands to all its Picture slots, such as `<Picture 2> and <Picture 3>`. Style numbering is independent of character, item and other role counters. Inspect Refpacks shows the exact mapping. Style references do not create additional Subject IDs or participate in Auto Outfit's wearer selection.

The role accepts images only, uses the existing RefPack file format and survives scene save/load. It guides visual style; it does not train a style LoRA or guarantee a particular rendering. Raw structured prompt passthrough does not expand `<Style n>`; use native `<Picture n>` tags there. The experimental Flux/LTX receivers do not run H3 alias expansion.

## How it works

Preparation stores cached VAE references plus the vision inputs needed by the native text encoder. Audio preparation stores native audio reference latents and timing metadata.

For a shot, the receiver:

1. Resolves group roles, native reference slots and stable Subject IDs.
2. Applies the video-reference window policy when generating video.
3. Compiles the instruction or passes through an allowed raw prompt.
4. Sends the cached visual inputs and prompt to the native H3 text encoder.
5. Attaches matching cached VAE reference data to the conditioning.
6. Allocates the output image/video latent layout for the connected model and selected generation mode.

Saving a pack avoids repeated **source VAE encoding**, but each shot still runs its prompt/text-vision encoding and the diffusion model still attends to references. A saved scene file uses the shared format and embeds all required data rather than linking to the individual asset files.

### Video-reference windowing

By default, a reference video longer than the output is limited to its leading H3-aligned window. Shorter references are not stretched. **Compress Motion** presents the full source reference instead, including for shorter output; it is a conditioning policy, not deterministic retiming or guaranteed choreography compression. Audio references are not cropped by this video-prefix policy.

For still output with the video architecture, complete input references remain available. The dedicated image architecture rejects video/audio reference entries.

## Compatibility

| Scenario | Status |
| --- | --- |
| Native MiniMax H3 Image model path | Implemented |
| Native H3 video/Ref2VA model path | Implemented |
| Video diffusion model with recognized image/still decoder | Implemented for still output |
| H3 Image diffusion model generating video | Rejected |
| p-h3-image | Not implemented |
| H3 still-image pack used on either native H3 model path | Implemented; preserves cached data without another VAE encode |
| Image, video and standalone audio references | Implemented where accepted by the receiving architecture |
| Character/Item/Motion aliases | Expanded only by the instruction compiler, not raw passthrough |
| Complete scene save/load | Supplied by the shared package |
| Standalone full-encode H3 RefMod import | Limited import; bundles and trained/pooled data are rejected |
| Other model-family latents | Not interchangeable; recreate with the appropriate VAE |

The current development checks comprise **101 passing CPU tests**, including native-source payload comparisons, saved-set fidelity, typed-role addressing, raw prompt passthrough and batch-wrapper composition. These checks do not establish pretrained-model image quality or performance. No live ComfyUI frontend or pretrained-model GPU generation was performed in that validation environment. See [VALIDATION.md](VALIDATION.md) and [CHANGELOG.md](CHANGELOG.md).

## Resolution and performance

- Output width and height must be at least 64 and divisible by 16.
- Reference resolution and output resolution are separate controls. Changing output dimensions does not re-encode a stored reference at a higher resolution.
- The H3 Image VAE preparation path uses separate native vision/VAE size helpers: 1024-short-edge vision inputs and a VAE short edge bounded between 1024 and 2048.
- `reference_megapixels` applies to the video-VAE preparation path; the image-VAE path uses those native size helpers.
- Reference video is resampled to 24 fps, aligned to `17n + 5`, and retains the native sampled vision frames and timestamps for text encoding. The video preparation node does not include source audio; prepare audio separately.
- Larger/more references still cost diffusion attention and memory after loading. No measured GPU speedup or minimum VRAM is claimed.
- Image batches require per-item native forward calls; requesting four images is additional generation work, not one free batched prediction.
- Upsizing or decoding an old reference cannot recover missing source detail. Small text, jewelry and other details can vary.

### Included workflows

| Workflow | Purpose |
| --- | --- |
| [01 H3 Dance From RefPacks](workflows/01_H3_Dance_From_RefPacks.json) | Character and motion reference generation with resolution, seed, duration and LoRA controls; current inspector and generation-mode sockets |
| [02 Prepare Character Or Outfit](workflows/02_Prepare_Character_Or_Outfit.json) | Still-image preparation and saving with a shared name input |
| [03 Prepare Dance Video](workflows/03_Prepare_Dance_Video.json) | Video preparation and saving |
| [04 H3 Image From RefPacks](workflows/04_H3_Image_From_RefPacks.json) | Image-model generation using saved references |
| [05 Prepare Audio](workflows/05_Prepare_Audio.json) | Standalone audio reference preparation |
| [06 H3 Character With Audio Reference](workflows/06_H3_Character_With_Audio_Reference.json) | Character, outfit and audio reference generation |
| [07 H3 General Picture References](workflows/07_H3_General_Picture_References.json) | General reference pictures without prescribed character roles |
| [08 Inspect Reference Mapping Only](workflows/08_Inspect_Reference_Mapping_Only.json) | Inspect slots and compiled prompt before encoding |

Weights and media are not bundled. Select locally installed files before running the examples.

## Troubleshooting

| Problem | Check |
| --- | --- |
| Shared package missing | Install the shared package using the registry command or one of the supported source folder names, then restart ComfyUI. |
| Nodes absent from search | Check startup errors, restart ComfyUI and refresh the browser. |
| Image VAE reports many latent frames | Select Image, or Auto with length 1–4, then connect the intended decoding VAE and route its output to VAEDecode. The sampler needs a single-frame output layout for an image VAE. |
| An image request becomes one image | In explicit Image mode, lengths above 4 deliberately fall back to one. Use 1–4 for the requested batch count. |
| Dedicated image model rejects a reference | Remove video/audio entries or use the video model architecture where appropriate. |
| Missing Character/Item/Motion/Picture tag | Inspect actual roles and slot numbering; ordinary instructions are validated against that map. |
| Compiled prompt rejected | Enable Allow Raw Prompt for structured text or JSON. |
| Outfit not applied automatically | Check that there is exactly one possible wearer, one outfit, Auto Outfit is on, and the prompt has no explicit outfit mention. Raw prompts bypass this behavior. |
| Wrong person, outfit, object or movement | Inspect reference order and compiled instructions, check for duplicate packs, then compare with known-working settings and the same seed. Roles do not guarantee exact control. |
| Newly saved pack missing | Refresh node definitions or the browser; individual packs and scene files use separate loaders. |
| Older RefMod bundle import rejected | Recreate from source media; the importer is for standalone full-encode entries. |
| Sampling is slow or runs out of memory | Check reference token sizes, source video length, output duration and active image batch count. |

## FAQ

### Is this a RefMod or a LoRA?

RefPacks has its own versioned container format. It stores reusable reference data rather than training weights or returning the earlier `H3_REF_MODS` objects.

### Does Qwen see the reference pictures?

Yes. This receiver supplies cached visual reference inputs to native H3 text encoding, alongside cached reference latents for diffusion. That differs from merely appending a latent after text encoding.

### Can the same still pack be used for images and video?

The implemented native H3 paths share still-image packs without another source VAE encode. That does not make them compatible with Flux, LTX or an arbitrary other model family.

### Can I generate a still with the video model?

Yes. Select Image, or Auto with length 1–4, and connect a matching H3 decoder. The model continues to use its native conditioning and stream layout.

### Can I save the entire cast, stage, outfit and motion setup?

Yes. Save the final references output with the shared Save RefPack Bundles node. Load that single file for each new shot and change the prompt or generation settings as needed.

### What if I want complete control of the prompt?

Enable Allow Raw Prompt and supply recognized structured text or JSON. The original text is passed unchanged; write native reference tags yourself.

### Does Motion guarantee exact movement transfer?

No. It gives a motion role and explicit addressing to a video reference. Actual motion adherence depends on the model, source and generation settings.

### Are all seeds guaranteed to preserve identity and details?

No. References and prompt structure guide generation; they do not guarantee identical details across all seeds.

## Reporting problems

Include:

- Shared package and H3 collection versions, plus ComfyUI version/commit.
- Exact diffusion checkpoint, text encoder, image/video VAE and audio VAE filenames in use.
- Workflow JSON and complete traceback for execution errors.
- Prompt, raw toggle, Auto Outfit, Compress Motion, seed, dimensions, length and sampling settings.
- The reference map, roles, source counts and whether inputs are image batches or sheets.
- Whether the set was assembled live or loaded from a saved scene file.
- Representative outputs for identity, outfit, item or motion failures.

Use [this collection's issues](https://github.com/BExplains/ComfyUI-MiniMax-H3-RefPacks/issues) for H3 behavior, and [shared RefPacks issues](https://github.com/BExplains/ComfyUI-RefPacks/issues) for shared persistence and reference-set problems. Remove private information from workflows before sharing them.

## Credits

- **[ComfyUI contributors](https://github.com/Comfy-Org/ComfyUI)** — native H3 implementation, VAE preprocessing, text encoding, multimodal conditioning and sampler infrastructure.
- **[MiniMax](https://github.com/MiniMax-AI)** — the H3 model family.
- **[Luisa / Luisacaotica](https://github.com/Luisacaotica)** — [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod), inspiration for the earlier RefMod work and the source format used by the limited importer.
- **[BExplains](https://github.com/BExplains)** — [shared RefPacks](https://github.com/BExplains/ComfyUI-RefPacks), this H3 collection, and the earlier [H3 Image RefMods creator](https://github.com/BExplains/ComfyUI-MiniMaxH3-Image-RefMods).

Neither the earlier creator nor ComfyUI-MiniMaxH3Mod is a runtime dependency of these RefPacks workflows. Attribution does not imply endorsement; dependencies and upstream projects retain their own licenses.

## License

Released under the [MIT License](LICENSE). Model weights and third-party dependencies retain their own licenses.
