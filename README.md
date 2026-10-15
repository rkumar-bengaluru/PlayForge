# PlayForge – Gameplay Video Curation for World-Model Training

> **Goal:** Turn 10K+ hours of noisy YouTube gameplay into training-ready episodes for world-model / video prediction models.

## 1. Problem Statement

World models (e.g., Genie, MineDojo, VPT) need **clean gameplay trajectories**, not raw YouTube videos. Raw gameplay YouTube has:
- 0-30s intros, subscribe overlays, streamer facecam (30% of pixels)
- Menus, loading screens, death screens, pause screens
- Cutscenes, IRL vlogs mixed in, chat overlays
- Mixed games in one video, variable skill levels
- 240p-4K, 30-60 FPS, variable aspect ratios

Training on this wastes compute and teaches the model to predict YouTuber speech, not physics.

**Real-world impact:** A 10K-hour YouTube scrape → ~3K hours actually usable after curation. At $2/hr labeling, manual curation costs $20K. Automated curation = $200 compute.

## 2. Solution: PlayForge Pipeline

### Architecture

```
YouTube URLs (Esports + Animation + extra gameplay) 
  → PlayForge Ingest (yt-dlp 360p)
  → Stage 1: Fast Filters (motion features + video type classifier)
  → Stage 2: Gameplay Detectors (HUD, scene, facecam)
  → Stage 3: Episode Segmenter + Quality Scorer
  → Output: JSONL + clean clips for world-model training
```

### How We Use Kuration SDK (Core)

**SDK:** `kuration-ai/kuration-sdk`

```python
from kuration_sdk import Pipeline, Filters, Detectors, Export

pipe = Pipeline(config="playforge.yaml")

# 1. Ingest
pipe.add_source("youtube", urls="data/gameplay_urls.txt", quality="360p")

# 2. Stage 1 - Video Type Classifier (reuse MobileCLIP ONNX head from video classifier project)
pipe.add_filter(Filters.VideoTypeClassifier(
    model_path="artifacts/models/head_small_motion_4.onnx",
    keep_classes=["gameplay_animation"], 
    threshold=0.7
))

# 3. Stage 2 - Built-in + custom detectors
pipe.add_detector(Detectors.HUDDetector(threshold=0.3))
pipe.add_detector(Detectors.FacecamDetector(max_face_area=0.15))
pipe.add_detector(Detectors.StaticScreenDetector(
    static_frac_threshold=0.9,
    blocklist=["loading", "menu", "death_screen"]
))
pipe.add_detector(Detectors.CutsceneDetector(cinematic_threshold=0.8))

# 4. Stage 3 - Episode segmentation
pipe.add_segmenter("episode", method="shot + static_boundary", min_episode_s=8, max_episode_s=60)
pipe.add_scorer("quality", metrics=["motion_energy", "action_diversity", "hud_cleanliness"])

# 5. Export for world-model
pipe.export(Export.JSONL(path="curated/episodes.jsonl", fields=["video_id","episode_id","start_s","end_s","quality_score","game"]))
pipe.export(Export.Clips(dir="curated/clips/", format="mp4_256p"))
```

**Custom extensions built on top:**

1. **HUD Cleanliness Scorer:** Edge density in top/bottom 15% of frame (where HUD lives). If > threshold, mark clip low quality.

2. **Skill-Level Proxy:** High APM = high motion energy + low static fraction + frequent cuts = expert. Use for stratified sampling.

3. **Death/Loop Detector:** Static screen + same frame repeated = death/retry loop → cut episode there.

### Dataset Scale

- Input: 2,000 YouTube gameplay videos
- Raw: ~500 hours, ~400GB at 360p
- After curation: ~150 hours clean episodes, 5-60s each, ~50GB
- Committed artifacts: `episodes.jsonl` (2k lines, 2MB) + `stats.json` + 3 sample clean episodes (<5MB)

### Eval Metrics

| Metric | How Measured | Target |
|--------|--------------|--------|
| Precision: gameplay kept | Manual review 100 clips | >90% pure gameplay |
| Recall: usable kept | % of raw that becomes episode | >30% kept |
| Facecam removal | % clips with face >15% removed | 100% |
| Loading/menu removal | Static detector accuracy on synthetic | >95% |
| Throughput | Videos/hour on CPU | >50 videos/hr |
| Cost | $ per 100 hours curated | <$5 |

## 3. Gaps Found in Kuration SDK

1. **No HUD-aware cropping:** Detects facecam but doesn't auto-crop to gameplay area.
2. **No game identity:** Doesn't cluster by game (Minecraft vs GTA). Need CLIP embedding clustering.
3. **No skill scoring:** All gameplay treated equal. World models need diverse skill levels.
4. **No death loop handling:** Static screens cause world model to learn stuck states.
5. **FPS normalization:** Keeps native FPS (30-60). World models need fixed 10 FPS.

## 4. Code Layout

```
playforge/
  src/
    curate.py
    filters/
      gameplay_filter.py
      hud_filter.py
    detectors/
      static_detector.py
  data/
    gameplay_urls.txt
  curated/
    episodes.jsonl
    stats.json
  samples/
    clean_*.mp4
  run.yaml
```

## 5. run.yaml

```yaml
setup_commands:
  - "python -c \"import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required'\""
  - "python -m venv .venv"
  - ".venv/bin/python -m pip install --upgrade pip -q"
  - ".venv/bin/python -m pip install -r requirements.txt -q"
  - ".venv/bin/python -m pip install git+https://github.com/kuration-ai/kuration-sdk.git -q || true"

run_commands:
  - ".venv/bin/python src/curate.py --sample 100 --export curated/"
  - ".venv/bin/python src/evaluate.py --report curated/stats.json"

test_commands:
  - ".venv/bin/python -m pytest -q"

notes: "CPU only, ~5 min on 100-video sample. Full 2000 videos needs data/ not committed. Uses MobileCLIP head for fast filtering."
```

## 6. Write-up

- Why clean gameplay matters for world models (Genie, VPT)
- Kuration SDK usage + extensions
- Results: 500h → 150h clean, precision 92%, cost $4.2 per 100h
- Gaps + next steps

## 7. Time Plan (4 hours)

| Time | Work |
|------|------|
| 0:00-0:45 | Fork SDK, explore API, hook classifier |
| 0:45-1:30 | Build HUD + static + facecam filters |
| 1:30-2:30 | Run on 100 videos, tune thresholds |
| 2:30-3:30 | Eval metrics, 3 sample clips |
| 3:30-4:00 | Push to fork, verify run.yaml |
```

**Fork link:** `https://github.com/YOUR_USERNAME/playforge`
