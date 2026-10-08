# Local audio generation models: research notes

Compiled 2026-10-08. Target machine: MacBook Pro, M2 Max, 32 GB unified memory.

**Corrected later the same day** by a closer look at two models, written up in
`2026-10-08-magenta-realtime-2-and-rave.md`. Where the two files disagree, that one is right:

- RAVE's code is CC BY-NC 4.0, not MIT.
- Magenta RealTime 2 does take an audio file as a style reference (in its Python library and
  its plugin, not in its `generate` command), and MIDI notes.
- Stable Audio 3's inpainting and LoRA training are confirmed in the local checkout; there is
  no separate "continuation" call, it is inpainting past the end of a clip.

## How to read this file

**Status tags**

- **[Verified]**: checked in the model's own repo or docs, or fetched directly.
- **[Secondary]**: from blogs, aggregators, directories, or a single third party. Treat as a lead.
- **[Unverified]**: a claim that conflicts across sources, or one I could not check.
- **[Not found]**: searched for, nothing usable.

**Input columns**

- **Text**: text prompt.
- **Audio ref**: an existing audio file steers or edits the output (timbre, style, continuation, cover, or inpainting).
- **Melody**: pitch or chroma of a reference only, not its timbre.
- **Video**: picture input.
- **MIDI**: symbolic output or input.

**Reviews**

- "User signal" is what users or reviewers said, sorted by how independent the source is.
- Most "reviews" found are SEO pages or developer claims. Reddit and Hacker News barely appear in the search index, and I did not open any Reddit pages directly.

---

## 1. Music generation (full songs, vocals, lyrics)

| Model | Inputs | Audio ref | M2 Max 32 GB | Licence | User signal | Status |
|---|---|---|---|---|---|---|
| **ACE-Step 1.5** (incl. XL 4B) | Text, lyrics | Yes: timbre reference; cover and repaint on a source file | XL needs ≥12 GB VRAM with offload, 20 GB recommended, so it fits in principle. Apple Silicon has no bf16, so a startup flag is needed. No M-series benchmark found. | Apache-2.0 | Mixed. One forum user found output "rather lackluster" and vocals failed. Blogs are enthusiastic (promotional). | Verified: licence, cover/repaint params (DiffSynth docs). Unverified: "beats Suno", SongEval 8.09 (arXiv 2602.00744). |
| **YuE** (original, Jan 2025) | Lyrics, genre tags | Yes: ICL checkpoint `YuE-s1-7B-anneal-en-icl` takes about 30 s of audio. Modes: mix, vocal, instrumental, dual-track. | Too heavy. Official repo: ≤24 GB handles 2 sessions; full songs want 80 GB. Quantised community builds run slower. | Earlier marketing said Apache-2.0. Unverified. | Early coverage: sample songs were mono. | Secondary |
| **YuE2** (Sept 2026) | Lyrics, tags | Unverified | A CivitAI user ran it on 16 GB. A Chinese report says 12 GB. A ComfyUI page says the release card targets 24 GB. | Weights reported as **CC BY-NC 4.0** (non-commercial) | CivitAI user: "really close to, or better than, Suno v5". MindStudio: Suno v6 still richer. Developer blind test: 57.3% vs 30.5% against Suno v4.5, and about a tie with Suno v5 (self-reported). | Secondary; hardware figures conflict |
| **DiffRhythm** (ASLP-lab) | Lyrics, style text | Optional audio prompt under 10 s | Base model about 8 GB VRAM (third-party). Local via Docker. | Conflicting: Apache-2.0 (AlternativeTo); "other" on the model card; VAE under Stability Community Licence | Setup friction. 49–57 open issues. A review says structure and vocals are weaker than instruments. | Secondary. Up to ~4:45. arXiv 2503.01183 (another source says 2502.05139). |
| **SongBloom** (Tencent AI Lab, NeurIPS 2025) | Lyrics | Yes: 10 s audio prompt (`prompt_wav`) | ComfyUI tutorial says ≥6 GB VRAM. The ComfyUI node is no longer supported. | Not checked | None found | Secondary. Up to ~2:30 stereo 48 kHz (paper's demo, plus aggregator). Code and weights on GitHub per the paper page. |
| **HeartMuLa** (`heartlib`) | Lyrics + tags | Not in the official pipeline. A community app (HeartMuLa-Studio) adds it. | Linux/CUDA only: Triton is not available on macOS. Skill doc says 8 GB with lazy load. A directory says 24 GB. | 3B: Apache-2.0 (Hugging Face) | Skill guide: tags are sometimes ignored, lyrics dominate, bf16 degrades HeartCodec (use fp32), dependencies conflict. | Verified: 3B licence. Developer claims a 7B internal version "comparable to Suno" (not released as far as found). |
| **Magenta RealTime 2** (Google, June 2026) | Text prompts (see below) | Not confirmed in the repo. Third-party write-up says text, audio and MIDI. | **Real-time on M2 Max** for `mrt2_base` (per the compatibility table). `mrt2_small` (230M) runs real-time on all Macs. Install with `uv pip install "magenta-rt[mlx]"` on Python 3.12. | Code Apache-2.0. Weights CC BY 4.0 per secondary sources. | No Mac hands-on found. Launch claim ~200 ms control latency. | **Verified** from repo and docs. Google's apps page says M3 Pro / M2 Max. The GitHub README says "Pro Max". |

**Notes**

- **Magenta RealTime 2**: `mrt2_base` is 2.4B, `mrt2_small` is 230M. Offline generation: `mrt mlx generate --duration N --model mrt2_base`. Offline generation works on any Apple Silicon Mac or NVIDIA GPU. Models download to `~/Documents/Magenta/magenta-rt-v2/`. Sources: [repo](https://github.com/magenta/magenta-realtime), [install docs](https://magenta.github.io/magenta-realtime/installation.html).
- **ACE-Step**: the 2B variant is listed as in progress in the `mlx-music` port (`pypi.org/project/mlx-music`; Python 3.11+, MLX 0.25.2+). Sources: [repo](https://github.com/ACE-Step/ACE-Step-1.5), [DiffSynth docs](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/ACE-Step.html), [Mac guide](https://lilting.ch/en/articles/ace-step-music-generation-mac), [Level1Techs thread](https://forum.level1techs.com/t/ace-step-1-5-journey-into-the-unknown/245674).
- **YuE2 vs Suno**: the developer blind-test figures are self-reported, and the critical reading notes they come from automatic metrics, not listening tests. Sources: [CivitAI](https://civarchive.com/articles/35173/yue2-is-a-wild-local-ai-generated-music-thats-really-close-to-or-even-better-than-suno-v5), [MindStudio](https://www.mindstudio.ai/blog/suno-v6-vs-yue2-music-generation), [Neurohive](https://neurohive.io/ru/papers/yue2-nejroset-dlya-sozdaniya-pesen/).
- **Other sources**: [YuE paper](https://arxiv.org/abs/2503.08638), [DiffRhythm listing](https://gittrend.io/repo/ASLP-lab/DiffRhythm), [SongBloom](https://www.runcomfy.com/comfyui-nodes/ComfyUI-SongBloom/song-bloom-generate), [HeartMuLa model card](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B), [HeartMuLa skill guide](https://skills.sh/fikriaf/agentos/heartmula).

---

## 2. Instrumental parts, stems, and loops

| Model | Inputs | Audio ref | M2 Max 32 GB | Licence | User signal | Status |
|---|---|---|---|---|---|---|
| **MusicGen-Stem** (Meta + IRCAM, arXiv 2501.01757) | Text; audio-conditioned stem generation | Yes: generates bass, drums and other stems conditioned on audio; edits stems in existing songs | Not verified. MusicGen itself is slow on CPU. | Weights `facebook/musicgen-stem-7cb` **CC-BY-NC 4.0** | None found | Secondary. **Code release not confirmed**: the STAGE paper said code and weights were not out as of April 2025. |
| **STAGE** (arXiv 2504.05690) | Audio context (full mix or click track) | Yes | Not verified | Not checked | None | Research. A fine-tune of MusicGen for stemmed accompaniment, with no extra context encoders. Code status unverified. |
| **StemGen** (ByteDance SAMI, arXiv 2312.08723) | Audio context | Yes: generates a stem that matches existing stems | Not verified | Not checked | None | Research. Code and weights **not confirmed**. |
| **Mustango** (AMAAI-Lab, NAACL 2024) | Text captions with explicit chords, beats, tempo, key | No | Not verified | Not checked. Built on Tango, so check that model's terms. | None found | Secondary. The paper says code and MusicBench are online. Checkpoint not found. |
| **Anticipatory Music Transformer** (Stanford CRFM) | MIDI prompts; infilling (see MIDI section) | No | CPU-friendly small models. Not benchmarked. | Apache-2.0 | See MIDI section | Verified: licence and checkpoints (per the repo README). |
| **ACE-Step 1.5 stems** (extract, lego / add track) | Audio | Yes | Unverified | Apache-2.0 | None | **Unverified**. Only third-party or hosted descriptions found; not in the official repo. |
| **Drum loop model** (four-bar, LooperMan / Freesound loop dataset; arXiv 2108.01576 benchmark) | Text, tempo | No | Docker local. Not verified. | Not checked | None | Secondary. Replicate demo `allenhung1025/looptest`. |
| **Stable Audio Open** (see §3) | Text | Fine-tuning on own audio is possible | Not verified | Community licence (conflicting) | Good for loops and one-shots. Not for full songs. | Secondary |
| **Stable Audio 3** (see §3) | Text | Yes: inpainting, continuation, audio-to-audio editing | Medium works locally (see §3) | See §3 | See §3 | See §3 |

**Notes**

- No model here was found to produce **seamless loops** (wrap-around generation). Workarounds to try: generate longer and crossfade, or inpaint across the loop seam with Stable Audio 3.
- Sources: [MusicGen-Stem paper](https://arxiv.org/abs/2501.01757), [STAGE paper](https://arxiv.org/html/2504.05690v2), [StemGen paper](https://arxiv.org/pdf/2312.08723v2.pdf), [Mustango paper](https://arxiv.org/abs/2311.08355), [AMT paper](https://arxiv.org/pdf/2306.08620), [Freesound loop benchmark](https://arxiv.org/pdf/2108.01576), [Replicate loop demo](https://replicate.com/allenhung1025/looptest).

---

## 3. Sound effects and textures (text to audio)

| Model | Inputs | Audio ref | M2 Max 32 GB | Licence | User signal | Status |
|---|---|---|---|---|---|---|
| **Stable Audio 3** (Stability AI, released 2026-05-20) | Text | **Yes**: inpainting, continuation, audio-to-audio editing. This project already uses it for variations. | **Medium runs locally** (you have it: 6.3 GB, tested). Third-party: a few seconds on an M4 MacBook Pro. Small variants are reported as CPU-capable. Large (2.7B) is API-only. | Code MIT (vLLM issue). Weights: Stability Community Licence (ComfyUI wiki). Check commercial terms. | Dubspot: "one of the more genuinely usable AI audio tools this year"; no vocals or lyrics. MindStudio: better long-form structure. One early skeptical HN comment about frequency range (old). | Secondary for features; your own test: on a pad, output moves toward the prompt and stays near the reference. |
| **Stable Audio 3 DAW plugin** | Text | — | AU and VST3 on Apple Silicon and Intel. AAX not listed. Beta. | — | Dubspot feature overview; not a session review | Secondary |
| **Stable Audio Open 1.0** | Text | Fine-tuning on own audio (no user results found) | 1.2B. Inference through `stable-audio-tools`. Text-only MLX port: `sandst1/stable-audio-mlx`. | Stability Community Licence (non-commercial per one source; conflicting) | Strong for loops, one-shots and ambient. Prompt detail matters. | Verified: 47 s max, 44.1 kHz (launch). Audio-to-audio not confirmed in sources. |
| **Stable Audio Open Small** | Text | Not confirmed | 341M; Arm-CPU oriented per one overview | Same as above | None | Unverified parameter count |
| **MAGNeT** (Meta, `audio-magnet-small` 300M, `audio-magnet-medium` 1.5B) | Text | No | Medium: 16 GB memory recommended on HF notes. Fits 32 GB. | Code MIT; **weights CC-BY-NC 4.0** | TTA-Bench: lowest perceptual quality of the four SFX models tested | Secondary |
| **AudioGen** (AudioCraft) | Text | No | MLX port (`mlx-audiocraft`) exists. | Not verified (AudioCraft code MIT; weights likely non-commercial) | None | Unverified |
| **TangoFlux** (declare-lab) | Text | No | 515M. 30 s at 44.1 kHz in 3.7 s on an A40. Setup guide asks for 6–8 GB RAM. Mac setup guide exists. | **Non-commercial research only** | Hobbyist praise; beats Stable Audio Open in one blog. FAD 2.26 vs 5.01 in a rival paper. | Secondary |
| **AudioLDM 2** | Text | No | Slow: 30 s+ for a 10 s clip. | CC-BY-NC(-SA) (sources conflict) | Benchmarks mixed: best objective quality in one table, lower subjective ratings | Secondary |
| **AudioLCM** | Text | No | Not verified | MIT (directory listing) | None | Unverified |
| **AudioX** (HKUSTAudio, arXiv 2503.10522) | Text, video, image, music, audio | Yes (multimodal input) | Checkpoints `AudioX`, `AudioX-MAF`, `AudioX-MAF-MMDiT`. ComfyUI node exists. | Gitee mirror: "not specified". Check the official repo. | None found | Secondary |
| **Woosh** (arXiv 2604.01929) | Text | Unknown | Not investigated | Unknown | None | Seen in search only; not investigated |
| **MiDashengLM-Gen** (arXiv 2608.11804) | Text | Unknown | Not investigated | Unknown | None | Seen in search only; not investigated |

**Notes**

- Stable Audio 3 sources: [chatforest review](https://chatforest.com/reviews/stability-ai-stable-audio-3-open-weight-music-sfx-generation/), [HF mirror](https://huggingface.co/niobures/stable-audio-3-medium), [Dubspot review](https://blog.dubspot.com/stable-audio-3-review), [MindStudio](https://www.mindstudio.ai/blog/what-is-stable-audio-3-stability-ai), [plugin overview](https://blog.dubspot.com/stable-audio-daw-plugin-2026). Paper arXiv 2605.17991.
- Stable Audio Open: [launch](https://stability.ai/news/introducing-stable-audio-open), [Arm ExecuTorch guide](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/run-stable-audio-with-executorch/2-download-model/).
- MAGNeT: [Hugging Face small](https://huggingface.co/facebook/audio-magnet-small), [paper arXiv 2401.04577](https://arxiv.org/pdf/2401.04577).
- TangoFlux: [Hugging Face](https://huggingface.co/declare-lab/TangoFlux), [Mac setup guide](https://codersera.com/blog/setting-up-tangoflux-for-text-to-audio-generation-on-mac/amp/).
- Benchmark: [TTA-Bench](https://arxiv.org/pdf/2509.02398).

---

## 4. Sound design and timbre tools

| Tool | Inputs | Audio ref | M2 Max 32 GB | Licence | User signal | Status |
|---|---|---|---|---|---|---|
| **RAVE** (IRCAM ACIDS, arXiv 2111.05011) | Audio. Trained on your own recordings. | **Yes**: timbre transfer and morphing | ~20× real time at 48 kHz on a laptop CPU. VST and nn~ (Max/Pd) builds exist; check the macOS download page. Training is GPU-heavy. | MIT (third-party listing). Verify on the repo. | r/MaxMSP: training slow without a dedicated GPU; people rent Colab T4s. IRCAM workshop: a composer built a singer's voice from RAVE models trained on a soprano's recordings. | Verified: 20× real time (IRCAM forum recap). Licence unverified. |
| **Gen-Synth** (RAVE + granular) | Audio | Yes | Max for Live device | Not checked | None | Research prototype; percussion-focused |
| **Magenta DDSP** | Audio | Yes (timbre transfer) | Real-time plugin prototype (2021) | Apache-2.0 (code) | Not checked | Secondary, older |
| **TexDSP / TexEnv** (DAFx 2025) | Noise and envelope parameters | No | PyTorch, lightweight | Open source per paper | None | Research. Sustained, noise-like textures. |
| **Neural proxies for synths** (Combes et al., arXiv 2509.07635) | Presets; audio for matching | Yes: sound matching to presets | Research code on GitHub | Not checked | None | Research |
| **Le Vaillant doctoral thesis** (UMONS) | Target audio | Yes: sound matching to synth presets | Synth-agnostic | Thesis | None | Research |
| **FM preset embeddings** (arXiv 2608.18226) | Presets; text and audio | Retrieval only | Not verified | Not checked | None | Research. 69% triplet agreement vs CLAP 72%. |
| **AI synth presets from text** | — | — | **Not found** | — | — | No open model found. LLM-prompted JSON presets (practitioner blog) only. |

**Notes**

- Sources: [RAVE paper](https://arxiv.org/pdf/2111.05011), [IRCAM tutorial](https://forum.ircam.fr/article/detail/tutorial-neural-synthesis-in-a-daw-with-rave/), [IRCAM forum recap](https://forum.ircam.fr/article/detail/retour-sur-notre-workshop-forum-au-festival-image-sonore-2024/), [r/MaxMSP thread](https://cal1.lr.ggtyler.dev/r/MaxMSP/comments/1is8cp3/rave_ircam_model_training), [TexDSP paper](https://dafx25.dii.univpm.it/wp-content/uploads/2025/07/DAFx25_paper_14.pdf), [neural proxies](https://arxiv.org/html/2509.07635v1), [Le Vaillant thesis](https://web.umons.ac.be/app/uploads/sites/6/2025/02/LeVaillantGwendal_Abstract-these_Vs2.pdf), [FM preset embeddings](https://arxiv.org/pdf/2608.18226).

---

## 5. Video to audio (foley)

| Model | Inputs | M2 Max 32 GB | Licence | User signal | Status |
|---|---|---|---|---|---|
| **MMAudio** (CVPR 2025, arXiv 2412.15322) | Video; optional text | Weights about 12 GB in total (large 44k is 4.12 GB). No Mac documentation found. | MIT | One hands-on test: keep prompts short and concrete. A host says ~157M params, fast on a GPU. | Verified: licence and weights (HF). Mac support not found. |
| **HunyuanVideo-Foley** (Tencent, arXiv 2508.16930) | Video + text | Project summary: ~20 GB VRAM, 24 GB recommended. Linux/CUDA. | Tencent Hunyuan Community Licence | A creator's ComfyUI walkthrough is positive; commenters hit import errors. | Secondary. Output 48 kHz stereo. |
| **ThinkSound** (FunAudioLLM) | Video + text; object-level refinement; NL-instructed editing of audio | Not verified | Code Apache-2.0. Model for research and education; commercial use needs contact. `thinksound.ckpt` ~950 MB incl. Synchformer. | None found | Secondary |
| **Kling-Foley** (arXiv 2506.19774) | Video | — | Public weights **not confirmed** | — | Not usable locally as far as found |

**Notes**

- Sources: [MMAudio HF](https://huggingface.co/hkchengrex/MMAudio), [Wiro MMAudio tests](https://wiro.ai/blog/mmaudio-4-video-to-audio-before-after-tests), [HunyuanVideo-Foley ComfyUI guide](https://aistudynow.com/hunyuanvideo-foley-comfyui-workflow-turn-quiet-video-into-sound/), [ThinkSound HF](https://huggingface.co/FunAudioLLM/ThinkSound), [Kling-Foley paper](https://arxiv.org/abs/2506.19774), [FoleyBench critique](https://ai.sony/publications/FoleyBench-A-Benchmark-For-Video-to-Audio-Models) (argues VGGSound-style datasets have poor audio-visual alignment).

---

## 6. Symbolic MIDI (song parts as notes)

| Model | Inputs | Output | M2 Max 32 GB | Licence | User signal | Status |
|---|---|---|---|---|---|---|
| **Anticipatory Music Transformer** (Stanford CRFM, arXiv 2306.08620) | MIDI; infilling (e.g. chords + drums → melody and bass) | MIDI | Small, medium and large checkpoints (`stanford-crfm/music-*-800k`). Runs on CPU. MIDInfinite web demo uses an MLC port. | Code Apache-2.0. Trained on Lakh MIDI; the paper has a copyright appendix. | Paper: human listeners rated accompaniments similar to human-composed music over 20 s. Stanford HAI (2023): not yet a seamless tool for musicians or sequencers. | Verified: licence, checkpoints, inference via Transformers and `events_to_midi`. No training code in repo. |
| **MIDI-LLM** (ISMIR 2026) | MIDI | MIDI | Not verified | Llama 3.2 Community Licence | None | Secondary |

**Notes**

- Sources: [paper](https://arxiv.org/pdf/2306.08620), [CRFM blog](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html), [Stanford HAI](https://hai.stanford.edu/news/composers-helper-using-ai-create-new-harmonies), [HF demo](https://huggingface.co/spaces/teamup-tech/anticipatory-music-transformer/blob/main/app.py).

---

## 7. Apple Silicon ports and tooling (MLX / Mac-native)

| Project | What it runs | Status | Licence | Notes |
|---|---|---|---|---|
| **Magenta RealTime 2** (`magenta-rt[mlx]`) | Real-time and offline music (text prompts) | Google's own; C++ engine on MLX | Apache-2.0 code; weights CC BY 4.0 (secondary) | Compiled `.mlxfn` container. Best Mac-native option here. |
| **mlx-music** (PyPI) | ACE-Step 1.5 (2B) | Early development; 2B listed as in progress | Not checked | Python 3.11+, MLX 0.25.2+ |
| **mlx-audiocraft** | MusicGen, AudioGen (MLX port of AudioCraft) | Listed as a 2026 port | Inherits AudioCraft licences (CC-BY-NC weights) | Source: a directory listing |
| **stable-audio-mlx** (`sandst1`) | Stable Audio Open Small (text-only) | Community port | Stability licence | Needs Hugging Face licence acceptance |
| **MLX-Audio** (Blaizzy) | TTS, STT, speech-to-speech; a music model | Established package | Not checked | Music model is MiniMax Music 3; not confirmed to run locally. Speech-focused. |
| **MusicGPT** (`gabotechs`) | MusicGen only | Precompiled macOS builds (Apple Silicon) | Not checked | Wrapper app |
| **MIDInfinite** | Anticipatory Music Transformer | Web demo; MLC port | Not checked | Local, commodity hardware |
| **Stable Audio 3 (this project)** | Medium, small variants | Running locally | See §3 | Used by this project's generator (`optimized/mlx`) |

**Notes**

- Sources: [mlx-music](https://pypi.org/project/mlx-music/), [mlx-audio](https://github.com/Blaizzy/mlx-audio), [stable-audio-mlx](https://github.com/sandst1/stable-audio-mlx), [MusicGPT](https://github.com/gabotechs/MusicGPT/blob/main/README.md), [Magenta RT2 install](https://magenta.github.io/magenta-realtime/installation.html).

---

## 8. Older and general-purpose models (background)

| Model | Inputs | Notes | Licence | Status |
|---|---|---|---|---|
| **Magenta RealTime** (v1, June 2025) | Text and audio style prompts, mixed | 800M parameters, 48 kHz stereo. Ran on Colab TPUs. Mac not confirmed. | Code Apache-2.0; weights CC BY 4.0 (secondary) | Superseded by RT2 |
| **MusicGen** (AudioCraft) | Text; melody variant takes a reference melody | 300M, 1.5B and 3.3B. Melody model is 1.5B. Mac: MLX port or MusicGPT. Old M1 test: about 60 s to make 9 s (500 steps). | Code MIT; weights CC-BY-NC 4.0 | Mixed reviews: a Max/MSP dev says melody following is reasonable but quality varies; a blogger praises it but says vocals are weak. |
| **Stable Audio Open 1.0** | See §3 | | | |

**Notes**

- Sources: [Magenta RT2 blog](https://magenta.withgoogle.com/magenta-realtime-2), [MusicGen paper](https://arxiv.org/abs/2306.05284), [MusicGen Mac guide](https://mustafa.net/2026/03/06/generate-music-locally-with-musicgen-no-subscriptions/).

---

## 9. Not usable locally, or not confirmed

| Model | Reason |
|---|---|
| **Suno, Udio** | Closed cloud services. Used only as comparison points in reviews. |
| **Lyria RealTime** (Google) | Closed. |
| **Stable Audio 3 Large** (2.7B) | API-only; no open weights. |
| **Kling-Foley** | Paper and benchmark only; no weights found. |
| **StemGen** (ByteDance) | Code and weights not confirmed released. |
| **MusicGen-Stem code** | Weights are on Hugging Face; official code not confirmed in the AudioCraft repo. |
| **AI synth presets from text** | No open model found. |
| **Seamless loop generation** | No model found that does wrap-around generation. |

---

## 10. Licence summary (for commercial use)

Check each licence before selling sounds. These are from sources that disagree in several cases.

| Licence type | Models |
|---|---|
| **Permissive (Apache-2.0 or MIT)** | ACE-Step 1.5 (Apache-2.0); Anticipatory Music Transformer (Apache-2.0); HeartMuLa 3B (Apache-2.0, per HF); DiffRhythm (Apache-2.0 per AlternativeTo, but "other" on the model card); RAVE (MIT per third-party listing, verify); MMAudio (MIT); AudioCraft code (MIT); MAGNeT code (MIT); Magenta RT2 code (Apache-2.0); ThinkSound code (Apache-2.0) |
| **Open weights, community or custom licence (check revenue and use terms)** | Stable Audio 3 (weights per Stability Community Licence; code MIT); Stable Audio Open (Stability Community Licence); HunyuanVideo-Foley (Tencent Hunyuan Community Licence); VAE in DiffRhythm (Stability Community Licence) |
| **Non-commercial (CC-BY-NC or research-only)** | MusicGen weights and MusicGen-Stem (CC-BY-NC 4.0); MAGNeT weights (CC-BY-NC 4.0); AudioLDM 2 (CC-BY-NC-SA or CC-BY-NC, conflicting); TangoFlux (non-commercial research); YuE2 weights (CC BY-NC 4.0, secondary); ThinkSound model (research and education) |
| **Not confirmed** | YuE (original); AudioGen weights; AudioX (Gitee mirror says "not specified"); Mustango; SongBloom; STAGE; Magenta RT2 weights (CC BY 4.0 per secondary) |
| **Llama licence** | MIDI-LLM (Llama 3.2 Community Licence) |

---

## 11. Open gaps and next steps

1. **User reviews**: Reddit and Hacker News barely show up in search. Next step is fetching specific threads or GitHub issues for the top candidates.
2. **Verify**: Magenta RealTime 2 audio and MIDI prompts; ACE-Step stems (extract, lego); Stable Audio 3 audio-to-audio parameters and licence file; RAVE licence; AudioGen licence; SongBloom and Mustango licences and checkpoints.
3. **Mac testing**: ACE-Step XL on MPS (speed, memory); Magenta RT2 `mrt2_base` on this M2 Max; RAVE training time on this machine; MMAudio on MPS.
4. **Not yet investigated**: Woosh (arXiv 2604.01929); MiDashengLM-Gen (arXiv 2608.11804); JukeDrummer (arXiv 2210.06007); Gen-Synth details.

---

## Source index

**Magenta RealTime**
- https://github.com/magenta/magenta-realtime
- https://magenta.github.io/magenta-realtime/installation.html
- https://magenta.withgoogle.com/magenta-realtime-2

**ACE-Step**
- https://github.com/ACE-Step/ACE-Step-1.5
- https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/ACE-Step.html
- https://lilting.ch/en/articles/ace-step-music-generation-mac
- https://forum.level1techs.com/t/ace-step-1-5-journey-into-the-unknown/245674
- https://www.promptspace.in/blog/ace-step-1-5-the-free-ai-music-generator-that-actually-beats-suno

**YuE and YuE2**
- https://arxiv.org/abs/2503.08638
- https://civarchive.com/articles/35173/yue2-is-a-wild-local-ai-generated-music-thats-really-close-to-or-even-better-than-suno-v5
- https://www.mindstudio.ai/blog/suno-v6-vs-yue2-music-generation

**DiffRhythm, SongBloom, HeartMuLa**
- https://gittrend.io/repo/ASLP-lab/DiffRhythm
- https://alternativeto.net/software/diffrhythm/about
- https://www.runcomfy.com/comfyui-nodes/ComfyUI-SongBloom/song-bloom-generate
- https://www.nextdiffusion.ai/tutorials/how-to-use-songbloom-to-make-ai-music-in-comfyui
- https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B
- https://skills.sh/fikriaf/agentos/heartmula

**Instrumental, stems, loops**
- https://arxiv.org/abs/2501.01757 (MusicGen-Stem)
- https://arxiv.org/html/2504.05690v2 (STAGE)
- https://arxiv.org/pdf/2312.08723v2.pdf (StemGen)
- https://arxiv.org/abs/2311.08355 (Mustango)
- https://arxiv.org/pdf/2108.01576 (loop benchmark)
- https://replicate.com/allenhung1025/looptest

**Stable Audio family**
- https://chatforest.com/reviews/stability-ai-stable-audio-3-open-weight-music-sfx-generation/
- https://huggingface.co/niobures/stable-audio-3-medium
- https://blog.dubspot.com/stable-audio-3-review
- https://blog.dubspot.com/stable-audio-daw-plugin-2026
- https://www.mindstudio.ai/blog/what-is-stable-audio-3-stability-ai
- https://stability.ai/news/introducing-stable-audio-open
- https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/run-stable-audio-with-executorch/2-download-model/
- https://github.com/sandst1/stable-audio-mlx

**Other SFX models**
- https://huggingface.co/facebook/audio-magnet-small
- https://arxiv.org/pdf/2401.04577 (MAGNeT)
- https://huggingface.co/declare-lab/TangoFlux
- https://codersera.com/blog/setting-up-tangoflux-for-text-to-audio-generation-on-mac/amp/
- https://arxiv.org/pdf/2509.02398 (TTA-Bench)
- https://arxiv.org/pdf/2604.01929 (Woosh, not investigated)
- https://arxiv.org/pdf/2608.11804 (MiDashengLM-Gen, not investigated)

**Sound design and timbre**
- https://arxiv.org/pdf/2111.05011 (RAVE)
- https://forum.ircam.fr/article/detail/tutorial-neural-synthesis-in-a-daw-with-rave/
- https://cal1.lr.ggtyler.dev/r/MaxMSP/comments/1is8cp3/rave_ircam_model_training
- https://dafx25.dii.univpm.it/wp-content/uploads/2025/07/DAFx25_paper_14.pdf (TexDSP)
- https://arxiv.org/html/2509.07635v1 (neural proxies)
- https://web.umons.ac.be/app/uploads/sites/6/2025/02/LeVaillantGwendal_Abstract-these_Vs2.pdf
- https://arxiv.org/pdf/2608.18226 (FM preset embeddings)

**Video to audio**
- https://huggingface.co/hkchengrex/MMAudio
- https://wiro.ai/blog/mmaudio-4-video-to-audio-before-after-tests
- https://aistudynow.com/hunyuanvideo-foley-comfyui-workflow-turn-quiet-video-into-sound/
- https://huggingface.co/FunAudioLLM/ThinkSound
- https://arxiv.org/abs/2506.19774 (Kling-Foley)

**MIDI**
- https://arxiv.org/pdf/2306.08620 (Anticipatory Music Transformer)
- https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html
- https://hai.stanford.edu/news/composers-helper-using-ai-create-new-harmonies

**MLX and Mac tooling**
- https://pypi.org/project/mlx-music/
- https://github.com/Blaizzy/mlx-audio
- https://github.com/gabotechs/MusicGPT/blob/main/README.md

**Background**
- https://arxiv.org/abs/2306.05284 (MusicGen)
- https://mustafa.net/2026/03/06/generate-music-locally-with-musicgen-no-subscriptions/
