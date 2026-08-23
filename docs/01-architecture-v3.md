# AWAS v3 — ML Core & Product Architecture

**Supersedes §3–§5 of [`00-master-plan.md`](00-master-plan.md).**
The thesis, evaluation protocol (§6), ethics (§8), and hardware analysis (§7)
of the master plan stand. This document replaces the *architecture*.

Written for a different bar: not "can this win a research competition" but
"can this become a company."

---

## 1. The honest diagnosis

v2 detects drowsiness with a hand-tuned geometric threshold (EAR → PERCLOS)
and responds with a template. Every part of that is replaceable in a weekend by
anyone with a webcam and MediaPipe.

Three specific weaknesses:

**W1 — The detector is a formula, not a model.** EAR is a 2D ratio between
landmark distances. It breaks under head yaw and pitch (foreshortening changes
the ratio with no change in eyelid position), it varies by a factor of two to
three across individuals, and it degrades badly with glasses reflections. A
fixed threshold across all drivers is the single largest source of both false
alarms and misses. This is a solved problem in the literature and v2 does not
use the solution.

**W2 — Nothing compounds.** Unit 1000 performs exactly like unit 1. No data
asset accumulates, no model improves, no driver gets a better experience for
having used it longer. A business needs something that gets better with scale;
v2 has nothing.

**W3 — The product is a detector, and detectors are commodity.** The market
already has good detectors. What the market does *not* have is a system that
knows whether its warnings actually worked, that predicts fatigue before it
arrives, or that adapts to the individual driver.

The good news: **v2's architecture is already pointed at the fix.** The
efficacy loop (§5.3) and the reflex/dialogue split (§4.4) are the two pieces
nobody else has. v3 makes them the centre of the system instead of a footnote.

---

## 2. Where the moat actually is

Five sources of defensibility, ordered by how hard they are to copy.

| # | Moat | Why it holds |
|---|---|---|
| **M1** | **Measured intervention efficacy** | Every DMS on the market fires an alert and forgets. None measure whether the driver actually recovered. AWAS records time-to-relapse per intervention type per driver — which turns intervention selection into a learnable problem and creates a dataset nobody else is collecting. |
| **M2** | **Voice as a fatigue channel** | AWAS is the only architecture in this category that *talks to the driver*, so it is the only one that can measure the reply. Response latency, speech rate, pause ratio, and articulation are validated fatigue correlates. A passive camera system structurally cannot do this. |
| **M3** | **Per-driver personalization** | A model that learns *your* baseline beats a population threshold, and gets better the longer you use it. That is both accuracy and switching cost: after three months AWAS knows this driver, and a competitor starts from zero. |
| **M4** | **Local data asset** | Public drowsiness datasets underrepresent Indonesian drivers, Indonesian cabin conditions, Indonesian vehicle types, and common local factors (headscarves, aftermarket window tint, tropical glare, night driving on unlit roads). A model tuned on that population is a real advantage over imported systems and cannot be bought. |
| **M5** | **Prediction, not detection** | Detection tells you the driver is already unsafe. Prediction tells a dispatcher at 22:00 that this driver on this route will likely be unsafe by 02:40. That is a different product sold to a different buyer at a different price. |

M1 and M2 fall directly out of v2's existing design. M3, M4, M5 are what this
document adds.

---

## 3. The ML core for drowsiness detection

This is the main ask. v2's pipeline was:

```
landmarks → EAR → threshold → PERCLOS → threshold → level
```

v3 replaces it with a five-tier stack. Each tier is independently testable and
independently shippable, which matters — see §13.

```
  frames
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ T0  GEOMETRY          face mesh, alignment, eye/mouth     │
│                       crops, head pose (PnP)              │
│                       ── deterministic, no learning       │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ T1  PER-FRAME MODELS  learned eye-state CNN, yawn CNN,    │
│                       gaze regressor                      │
│                       ── replaces EAR/MAR thresholds      │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ T2  TEMPORAL MODEL    TCN over 60–120 s of per-frame      │
│                       features → level logits + variance  │
│                       ── replaces the PERCLOS window      │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ T3  PERSONALIZATION   per-driver baseline, z-scoring,     │
│                       driver embedding, on-device adapt   │
│                       ── new in v3 (moat M3)              │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ T4  MULTIMODAL FUSION face + vehicle + voice, late fusion │
│                       with calibrated uncertainty         │
│                       ── v2 had face+vehicle, adds voice  │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ T5  PREDICTION HEAD   hazard model: P(unsafe within Δt)   │
│                       ── new in v3 (moat M5)              │
└──────────────────────────────────────────────────────────┘
```

### T0 — Geometry (deterministic)

Unchanged from v2 and still useful: face detection, 478-point mesh, similarity
alignment, eye and mouth crops, head pose via PnP.

**Key change in role.** T0's outputs are now *inputs to learned models*, not
decision rules. EAR and MAR survive as features, not as thresholds.

### T1 — Learned per-frame state

| Model | Input | Output | Size |
|---|---|---|---|
| Eye state | 32×32 aligned grayscale eye crop, per eye | P(closed), openness ratio | ~50 k params |
| Yawn | 64×64 mouth crop | P(yawning) | ~50 k params |
| Gaze | eye crops + head pose | yaw/pitch of gaze, on-road/off-road | ~80 k params |

**Why this beats EAR.** The classifier sees pixels, so it learns eyelid
appearance directly rather than inferring it from a pose-sensitive ratio. It
handles glasses, partial occlusion, squinting, and — critically — individual
and population variation in eye shape that a single geometric threshold cannot
accommodate. This is the fix for W1.

**Training data:** MRL Eye Dataset, CEW, ZJU eyeblink, plus own collection
(§5.1). These are small models on small crops. The implemented version is
23.5k parameters and, exported to int8 ONNX, runs both eyes in **0.11 ms
single-threaded on an M1 Pro** (measured, `training/README.md`). A Pi 5 is
several times slower — measure on the device before this number enters the
latency budget.

**Ship the honest baseline too.** Keep EAR as a fallback and as the control
condition. If the learned model does not beat it on held-out Indonesian faces,
that is a finding worth reporting, not a failure to hide.

### T2 — Temporal model

PERCLOS's weakness is that it is a *summary statistic over a fixed window*: it
throws away the shape and ordering of events. Blink-duration lengthening,
closure-curve asymmetry, and nod-then-recover signatures all carry information
that a percentage destroys.

- **Input:** per-frame feature vector (eye probs, openness, yawn prob, gaze,
  head pose, and their deltas) at 30 Hz, downsampled to 5 Hz, over a 120 s
  causal window.
- **Model:** temporal convolutional network — dilated causal convolutions,
  ~300 k params. Chosen over LSTM/GRU because it is parallelizable, has fixed
  memory, and carries no recurrent state to corrupt when frames drop, which
  they will.
- **Output:** logits over levels 0–4 **plus a variance estimate**, so
  `confidence` becomes a calibrated quantity rather than a heuristic. The
  gating logic in master plan §5.2 already keys off confidence — this makes
  that gate mean something.

**Benchmark the simple thing first.** Gradient-boosted trees over the v2
windowed features (PERCLOS, microsleep count, blink duration, yawn rate) are
often competitive with sequence models on this task and cost a fraction as
much. Run GBT as the baseline. **If GBT wins, ship GBT.** Deep learning that
does not beat a boosted tree is a liability on an embedded device, not an
achievement.

### T3 — Personalization (new — moat M3)

The largest accuracy gain available, and the one that compounds.

**Baseline estimation — built, and it needs no model.** The no-ML portion of
this tier is implemented in `awas/driver/calibration.py`: robust quantiles of
EAR accumulated only during *known-alert* periods give each driver an `open_ref`
and a `closed_ref`, and closure is scored on that personal scale rather than
against a constant. Profiles persist between trips and merge weighted by sample
count, with a cap so an established profile still moves when glasses go on or
the camera is knocked. Maturity drives `confidence`, which the state machine
already uses to downgrade an under-evidenced level — so a driver the system has
just met gets a quieter system, by construction.

Blink rate, blink duration and neutral head pose extend the same mechanism and
are the natural next additions. Known-alert is defined conservatively:
within the first 10 minutes of a trip, following a rest gap, with low lane-
position variance and no drowsiness-consistent features.

**Deviation features.** Feed the temporal model z-scores against the driver's
own distribution, not raw values. "This driver's eyes are 2.3 MAD below their
own alert baseline" is a far stronger signal than "this driver's EAR is 0.19."

**Driver embedding.** A 32-dimensional learned vector per driver, concatenated
into T2's input. Updated on-device, stored on-device, never transmitted as raw
data. This is what lets the model specialize without retraining from scratch.

**Cold start.** New drivers use the population prior, and `confidence` reflects
that explicitly. The system is honest about knowing less on day one, and the
gating logic already knows what to do with low confidence.

**This is the switching cost.** Three months of accumulated personal baseline
is not portable to a competitor's product.

### T4 — Multimodal fusion

v2 fused facial and vehicle evidence with hand-set weights. v3 makes it a
learned late fusion over three channels, each contributing an estimate *and* an
uncertainty:

| Channel | Signals | Property |
|---|---|---|
| Facial | T3 output | Highest resolution, fails on occlusion |
| Vehicle | SDLP, steering reversal rate, steering entropy, speed variance | Degrades earlier than facial; survives occlusion; needs a road camera or OBD |
| **Voice** | response latency, speech rate, pause ratio, pitch variance, articulation | Sparse (only during dialogue) but high-value; **unique to AWAS** (§4) |

Fusion is uncertainty-weighted, so a channel that is unavailable or unreliable
degrades gracefully instead of poisoning the estimate. This preserves master
plan invariant I4 and makes the degradation ladder (§5.4) principled rather
than a table of special cases.

### T5 — Prediction head (new — moat M5)

**The product-defining tier.** Detection says "you are unsafe now." Prediction
says "you will likely be unsafe in 20 minutes, and the last rest stop before
then is in 12."

- **Framing:** discrete-time hazard model — P(level ≥ 3 within the next Δt |
  history), for Δt ∈ {15, 30, 60} minutes.
- **Features:** circadian phase (time-of-day as sine/cosine — the 02:00–06:00
  and 14:00–16:00 troughs are the strongest single predictors in the fatigue
  literature), time-on-task, cumulative microsleeps this trip, trip start time,
  gap since previous trip end (a sleep-opportunity proxy), route monotony from
  GPS, current feature trajectory slopes, and this driver's own historical
  onset pattern.
- **Model:** gradient-boosted survival model, or a survival head on T2.
- **Output:** a risk curve over the next 60 minutes, not a binary.

**Be honest about difficulty.** Prediction needs longitudinal data across many
drivers and many trips, which means it cannot be validated until deployment is
underway. It is Phase 3, not Milestone 1 — but the *logging schema* for it must
exist from day one, or the data will not be there when it is needed.

---

## 4. Voice as a fatigue channel (moat M2)

The strongest idea in this document, and it falls out of v2's existing design
for free.

AWAS already talks to the driver and already prompts for verbal replies at
escalation level 2 and above. That means AWAS already has:

- a microphone that is already open,
- a prompt with a known issue time,
- an expected response.

From which the following are directly measurable:

| Signal | Extraction | Fatigue relation |
|---|---|---|
| Response latency | prompt timestamp → voice-activity onset | Lengthens with fatigue |
| Non-response | timeout | Strong danger signal |
| Speech rate | syllables or phonemes per second, from ASR timing | Slows with fatigue |
| Pause ratio | silence fraction within the utterance | Rises with fatigue |
| Pitch variance | F0 statistics | Flattens with fatigue |
| Articulation | ASR confidence, phoneme boundary sharpness | Degrades with fatigue |
| Content coherence | reply relevance to the prompt | Degrades late — high specificity |

**Why this is defensible.** A passive DMS cannot collect any of it. Seeing
Machines and Netradyne cannot bolt this on without becoming conversational
systems, which is an architectural change, not a feature. AWAS gets it as a
byproduct of a design decision it already made for a different reason.

**Cost:** one VAD, timing instrumentation around the existing dialogue path, and
statistics over the ASR output that Milestone 4 already produces. There is no
new hardware and no new model.

**Risks:** cabin noise, passengers speaking, music, and the ethics of always-on
audio. Audio is processed in a short rolling buffer and never persisted, on the
same terms as video under master plan §8, and the driver is told this plainly.

---

## 5. The learning system

Models are the easy part. The system that produces labels and improves models
in the field is the actual engineering.

### 5.1 Label acquisition — the real bottleneck

Ordered by cost per label, cheapest first:

| Source | Quality | Cost | Notes |
|---|---|---|---|
| Public datasets | Good, wrong population | Free, slow to obtain | NTHU-DDD, UTA-RLDD, DROZY, MRL, CEW, YawDD — several need signed agreements, **start now** |
| Proxy: microsleep events | High precision, low recall | Free | A 2 s eye closure at 80 km/h is unambiguous |
| Proxy: intervention outcome | Noisy but abundant | Free | Relapse, pull-over, response latency |
| **Self-report (KSS)** | Gold standard subjective | Nearly free | **The companion can simply ask.** Again unique — a passive system has no channel for this |
| Active learning upload | High | Consent + bandwidth | Upload only *uncertain* feature sequences for review, never routine ones |
| Own collection | High, right population | Expensive | Simulator + parked vehicle, Indonesian drivers (moat M4) |

The KSS point is worth dwelling on: AWAS can ask "on a scale of one to nine,
how alert are you feeling?" at trip start and end, conversationally, and get
labelled data as a natural part of the interaction. That is a structural
labelling advantage.

### 5.2 On-device adaptation, off-device aggregation

The privacy invariant (master plan I5: raw video never leaves the device) is
not an obstacle to the data moat — it is the shape of it.

```
   ┌─── device ────────────────────────────┐
   │  frames ──► features ──► models       │   raw video: never leaves,
   │                │                      │   never written to disk
   │                ▼                      │
   │  personal baseline + driver embedding │   stays local
   │                │                      │
   │                ▼                      │
   │  feature sequences + weak labels      │   local ring buffer
   │                │                      │
   │        (consent, uncertainty gate)    │
   └────────────────┼──────────────────────┘
                    ▼
   ┌─── platform ──────────────────────────┐
   │  secure aggregation + DP noise        │
   │  global model training                │
   │  fleet analytics, risk maps           │
   └───────────────────────────────────────┘
```

- Raw video and raw audio: never transmitted, never persisted.
- Personal baselines and driver embeddings: never transmitted.
- Derived feature sequences: transmitted only when the active-learning gate
  fires and consent covers it.
- Model improvements: aggregate, with secure aggregation and differential
  privacy noise.

This is a genuine differentiator in a market where the incumbent fleet products
upload driver video to the cloud by default, and it matters legally — under
Indonesia's Personal Data Protection Law (UU No. 27/2022), biometric data is
sensitive personal data with correspondingly heavier obligations. *Verify the
current implementing regulations with counsel before any commercial pilot; do
not rely on this document for legal conclusions.*

### 5.3 Intervention policy learning (moat M1)

v2 selects interventions with a fixed ladder. v3 learns the selection.

- **Framing:** contextual bandit, not full RL. The context is driving state,
  level, workload, time of day, and driver embedding. The arms are intervention
  variants — template family, phrasing register, voice, timing. The reward is
  time-to-relapse, adjusted for response latency and an annoyance proxy
  (dismissals, driver mute requests).
- **Algorithm:** Thompson sampling over a small arm set. Not deep RL — the data
  volume is not there and the safety envelope does not permit it.
- **Hard safety boundary:** the bandit chooses **how** to intervene, never
  **whether**. The state machine's gating (master plan §5.2) and the reflex path
  (invariant I1) are outside the learned system entirely. A learned policy may
  never suppress a level-4 alert, delay a reflex response, or alter escalation
  at levels 3 and 4.

This is the loop that makes the product improve with usage, and it is the
dataset nobody else is collecting.

---

## 6. Feature surface beyond drowsiness

Fleets do not buy a drowsiness detector; they buy risk reduction. Additional
capability, roughly by effort against the same pipeline:

| Feature | Effort | Buyer value |
|---|---|---|
| Phone use, smoking, eating, seatbelt, driver absent | Low — same crops, same models | Table stakes for fleet sales; commonly requested |
| Post-trip driver coaching (non-punitive) | Low | Behaviour change, insurance evidence |
| Fleet fatigue dashboard: risk by driver, route, shift | Medium — server side | **This is the recurring revenue** |
| Shift-scheduling recommendations from T5 | Medium | Operational, sticky, hard to rip out |
| Route risk mapping from aggregated events | Medium | Compounding data asset (M4) |
| Insurance telematics integration | Medium — partnership | Classic monetization wedge; premium discount funds the unit |
| Sudden-incapacitation detection (medical event) | High | Very high value, rare, needs careful FP handling |
| rPPG — pulse from facial video | Research | Fragile under vehicle motion and variable light; do not commit to it |

The fleet dashboard row is the one that turns this from a device into a
company. A box sold once is hardware margin; a fleet paying monthly for fatigue
risk management is a business.

---

## 7. Deployment tiers — and the go-to-market problem

**The Rp5.6M BOM is a serious go-to-market obstacle**, and v2 does not
acknowledge it. That is roughly USD 350 per vehicle, up front, for an operator
whose margin per vehicle is thin. For an individual driver or a small operator
running a handful of vehicles, it is simply out of reach — and those are
exactly the operators the master plan's national-scale argument depends on.

| Tier | Form | BOM | Reaches |
|---|---|---|---|
| **A — AWAS Mobile** | Android app, phone in a windshield mount | **Rp0** | Individual drivers, small operators, pilots |
| **B — AWAS Box** | Pi 5 retrofit (master plan §7) | ~Rp5.6M | Fleets wanting fixed, tamper-resistant install |
| **C — AWAS Embedded** | Module for OEM / bodybuilder integration | — | Later; requires scale first |

**Tier A is strategically the important one**, for three reasons:

1. **Zero marginal cost.** Distribution through the Play Store or fleet MDM,
   not through hardware logistics and installation labour.
2. **Mid-range Android NPUs are competitive with — often better than — a Pi 5
   CPU** for exactly this workload. The compute argument for the box is weaker
   than it looks.
3. **It fills the data moat fastest.** M3 and M4 need drivers and trips, and
   the cheapest path to thousands of trips is an app, not a supply chain.

Tier A limitations, stated plainly: no OBD-II, no forward road camera (so no
SDLP and no traffic-light context — the vehicle channel of T4 falls back to GPS
only), phone thermal throttling on long trips, the phone is needed for other
things, and a mount is still required.

**The vehicle channel is where the box earns its price.** Tier A is a strong
driver-monitoring product; Tier B is the full three-layer system. Sell them as
what they are rather than pretending they are equivalent.

---

## 8. Revised system architecture

```
┌──────────────────────── EDGE (phone or box) ─────────────────────────┐
│                                                                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                          │
│  │ driver cam│  │ road cam  │  │ mic       │  ┌──────────┐            │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  │ OBD/GPS  │            │
│        │              │              │        └────┬─────┘            │
│        ▼              ▼              ▼             ▼                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐  ┌──────────┐             │
│  │ T0 geom  │   │ lane/    │   │ VAD +    │  │ speed,   │             │
│  │ T1 frame │   │ light/   │   │ ASR      │  │ brake,   │             │
│  │ T2 tempo │   │ distance │   │          │  │ position │             │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘  └────┬─────┘             │
│       │              │              │             │                   │
│       ▼              │              ▼             │                   │
│  ┌──────────┐        │        ┌──────────┐        │                   │
│  │ T3       │        │        │ voice    │        │                   │
│  │ personal │        │        │ biomarker│        │                   │
│  └────┬─────┘        │        └────┬─────┘        │                   │
│       │              │             │              │                   │
│       └──────────────┴─────┬───────┴──────────────┘                   │
│                            ▼                                          │
│              ┌──────────────────────────┐                             │
│              │ T4 FUSION                │                             │
│              │ level + calibrated conf  │                             │
│              │ + workload               │                             │
│              └───────────┬──────────────┘                             │
│                          │                                            │
│              ┌───────────┴──────────────┐                             │
│              ▼                          ▼                             │
│    ┌──────────────────┐      ┌──────────────────┐                     │
│    │ STATE MACHINE    │      │ T5 PREDICTION    │                     │
│    │ pure · gating    │      │ hazard curve     │                     │
│    │ SAFETY ENVELOPE  │      └────────┬─────────┘                     │
│    └────┬───────┬─────┘               │                               │
│         │       │                     │                               │
│    reflex│       │dialogue            │                               │
│         ▼       ▼                     │                               │
│  ┌─────────┐ ┌──────────────────┐     │                               │
│  │ pre-    │ │ POLICY (bandit)  │◄────┘                               │
│  │ rendered│ │ chooses HOW      │                                     │
│  │ <300 ms │ │ never WHETHER    │                                     │
│  └─────────┘ └────────┬─────────┘                                     │
│                       ▼                                               │
│              ┌──────────────────┐                                     │
│              │ companion + TTS  │                                     │
│              └────────┬─────────┘                                     │
│                       ▼                                               │
│              ┌──────────────────┐                                     │
│              │ EFFICACY MONITOR │──► local label store                │
│              │ time-to-relapse  │                                     │
│              └──────────────────┘                                     │
└───────────────────────────┬───────────────────────────────────────────┘
                            │  features + weak labels only
                            │  (consent + uncertainty gated)
                            ▼
┌──────────────────────── PLATFORM ─────────────────────────────────────┐
│  secure aggregation · global training · fleet dashboard · risk maps   │
│  scheduling recommendations · insurance integration                    │
└───────────────────────────────────────────────────────────────────────┘
```

**The safety envelope is the important structural property.** Everything
learned — T1 through T5 and the policy bandit — sits *outside* the state
machine. The state machine stays pure, deterministic, auditable, and testable,
and it retains final authority over whether an intervention fires. Learned
components propose; the safety envelope disposes. Master plan invariants I1–I5
survive intact, which is what makes it defensible to put ML in a safety product
at all.

---

## 9. Code structure implications

```
awas/
  core/          state machine — UNCHANGED, stays pure (I2)
  perception/
    geometry/    T0 — mesh, alignment, crops, PnP
    frame/       T1 — learned per-frame models
    temporal/    T2 — sequence model + calibration
  personal/      T3 — baselines, z-scoring, driver embedding, on-device adapt
  road/          Layer 2 — lane, SDLP, light, distance
  voice/         voice biomarker extraction (rides on speech/)
  fusion/        T4 — uncertainty-weighted late fusion
  predict/       T5 — hazard model
  policy/        intervention bandit — proposes only, never gates
  companion/     dialogue manager, templates, LLM
  speech/        TTS out, VAD + ASR in
  sources/       mock, webcam, phone, OBD, GPS, simulator
  runtime/       loop, efficacy monitor, session logging
  learning/      label store, active-learning gate, federated client

platform/        separate service — aggregation, training, fleet dashboard
```

`core/` does not change. That is deliberate and it is the point: 40 passing
tests keep passing, and the safety-critical layer is unaffected by everything
above.

---

## 10. Competitive honesty

Worth writing down so we do not fool ourselves or a judge.

**Not differentiated — table stakes, must work, will not win anything:**
PERCLOS-style drowsiness detection, phone-use detection, lane departure
warning, yawn detection, basic dashcam capture.

**Differentiated:**
measured intervention efficacy (M1), voice-channel fatigue sensing (M2),
per-driver personalization (M3), Indonesian-population data asset (M4),
fatigue prediction (M5), and the phone-tier price point (§7).

**Incumbents:** Seeing Machines and Smart Eye hold OEM positions; Netradyne,
Samsara, Motive, and Nauto hold Western fleet telematics. None of them is
optimized for the Indonesian market, none does closed-loop efficacy
measurement, and none has a conversational channel. That is the gap.

**Regulatory tailwind:** the EU General Safety Regulation mandates driver
drowsiness and attention warning systems on new vehicles, with distraction
warning following. Regulation of this kind tends to propagate to other markets
over time. *Verify current scope and dates before putting them in a pitch —
this document should not be cited as the source.*

---

## 11. Staging — so Milestone 1 still ships

One caution, stated once. This architecture is considerably larger than v2, and
the competition deadline has not moved. The tiers are ordered so that the
system is useful and demonstrable at every stage, and **nothing below Phase 1
is required for the Milestone 1 laptop demo.**

| Phase | Content | Gets you |
|---|---|---|
| **1 — now** | T0 + T1 + GBT baseline at T2, reflex path, template dialogue, efficacy **logging schema** | The Milestone 1 demo, with a real learned detector instead of a threshold |
| **2** | T2 sequence model, T3 personalization, voice biomarkers, KSS self-report | The research contribution and the first two moats |
| **3** | T4 full fusion, policy bandit, fleet dashboard, Tier A phone app | The product |
| **4** | T5 prediction, federated aggregation, insurance integration | The company |

**The one thing that must be right from day one is the logging schema.** T5 and
the policy bandit cannot be trained on data that was never recorded. Design the
event log in Phase 1 for what Phase 4 will need — it costs almost nothing now
and is unrecoverable later.

---

## 12. Risks

| Risk | Response |
|---|---|
| Scope kills the competition deliverable | Phase 1 is self-contained; everything else is explicitly deferred (§11) |
| Learned models do not beat EAR/GBT baselines | Then ship the baseline and report it. The moats (M1, M2, M3) do not depend on deep learning winning |
| Label bottleneck stalls T1/T2 | Public datasets first, KSS self-report second, own collection third — all three run in parallel |
| Prediction (T5) never validates | It is Phase 4 and gated on deployment data. The product stands without it |
| Incumbents add a conversational channel | Architectural change for them, and M1's efficacy dataset is already accumulating by then |
| Biometric data regulation | Privacy-by-architecture is already the design (§5.2); get counsel before any commercial pilot |
| ML in a safety product | The safety envelope (§8) keeps every learned component outside the gating path |
| Phone tier cannibalizes box sales | They serve different buyers; price and position them as different products (§7) |

---

## 13. Open questions

1. **Tier A first, or Tier B first?** Data moat argues phone-first; the research
   competition and the existing plan argue box-first. They are not exclusive but
   they compete for the same weeks.
2. Which incumbent is the realistic first competitor in Indonesia — is any
   fleet telematics vendor already selling DMS here?
3. Is the first buyer a fleet operator, an insurer, or a logistics platform?
   The insurer path funds the hardware but lengthens the sales cycle.
4. Own-data collection: how many Indonesian drivers can realistically be
   recruited, and through what channel?
5. Does the competition permit or reward a commercialization track? Some do
   explicitly; it changes how much of this belongs in the submission.
