# AWAS v2 — Master Plan

**Advanced Warning & Awareness System**
Research competition project, 2026 · Targhib & Fachry

> This plan supersedes the original project brief. It keeps that document's
> thesis intact and rebuilds the engineering and research design around it.
>
> **Sections 3-5 (architecture, layer specs, intervention controller) are
> superseded by [`01-architecture-v3.md`](01-architecture-v3.md)**, which
> replaces the threshold-based detector with a learned ML stack and adds the
> personalization, prediction, and policy-learning layers. Sections 6-10
> (evaluation, hardware, ethics, roadmap, risks) still stand.

---

## 1. What changed from v1, and why

The original brief makes a strong argument. Its weaknesses are not in the idea —
they are in the parts a research competition actually judges: measurement,
scope realism, and failure analysis. Here is every change, with the reason.

| # | Change | Why |
|---|---|---|
| 1 | **Added a full evaluation protocol** (§6) — research questions, hypotheses, baselines, metrics, datasets, statistics | v1 had none. A judge cannot distinguish "we built a thing" from "we showed the thing works." This is the single biggest gap. |
| 2 | **Added vehicle-behaviour signals to drowsiness detection** (§4.3) | Lane-position variability and steering entropy are validated drowsiness indicators. Fusing them with facial signals is a *defensible research contribution*; face-only detection is a solved commodity. |
| 3 | **Scoped Layer 2 down** — dropped pedestrian and general hazard detection (§4.2) | v1 asks a Pi to run driver CV + road CV + an LLM at once. It cannot. v1's own text says Layer 2 exists only to supply context, so we build only the context that changes an intervention decision. |
| 4 | **Split the intervention path in two: reflex vs. dialogue** (§5.1) | An LLM cannot be on a safety-critical path — its latency is unbounded and its output is unverified. Emergencies get pre-rendered audio in <300 ms; conversation gets up to 3 s. This becomes a hard architectural invariant. |
| 5 | **Specified drowsiness levels concretely** as PERCLOS + microsleep + head-pose with hysteresis (§4.1) | v1 says "detect eye closure duration, blink frequency, head position" but never defines the output. The prototype already assumes a 0–4 level with nothing producing it. |
| 6 | **Added a closed feedback loop** — measure whether an intervention worked, adapt (§5.3) | Turns the *product* claim ("it keeps drivers alert") into a *measurable* claim, and makes the system genuinely adaptive rather than just conversational. |
| 7 | **Added latency, thermal, and power budgets** (§7.2) | A real-time safety system with no timing analysis is a claim, not a design. |
| 8 | **Revised hardware: Pi 4B → Pi 5 + active cooling + NVMe** (§7.1) | Sustained CV + LLM inference thermally throttles an uncooled Pi within minutes. Also fixes two BOM defects (see §7.3). |
| 9 | **Added a graceful-degradation ladder** (§5.4) | The system must stay safe when the LLM is slow, the camera is blinded by sun, or OBD-II is absent. v1 assumes everything works. |
| 10 | **Added privacy, ethics, and research-safety policy** (§8) | An always-on camera pointed at a person's face. No on-road drowsy-driving experiments. Both need explicit written policy for a competition. |
| 11 | **Made the "vehicle-agnostic" claim honest** (§7.4) | Buses and trucks use SAE J1939, not OBD-II; many older Indonesian vehicles expose speed unreliably. GPS fallback is required for the national-scale claim to hold. |

**Unchanged from v1:** the core thesis, the three-layer framing, the
offline/edge requirement, retrofit affordability, and the target of public and
commercial fleets. Those are good and we are not touching them.

---

## 2. Thesis

Most road accidents in Indonesia are not caused by vehicle failure but by
**decline in driver cognitive readiness** — drowsiness, distraction, delayed
response. Existing systems *detect* this and then emit a passive alarm that
drivers habituate to and ignore. Nothing on the market *manages* driver state
as a continuous, closed-loop part of the safety system.

AWAS treats driving safety as a **human cognitive-system problem**, not a
vehicle-sensor problem. It fuses driver state with road context and intervenes
with adaptive conversation — a safety control mechanism, not entertainment —
entirely offline, retrofit, and vehicle-agnostic.

### The one-sentence research claim

> Conversational intervention, gated by fused driver-state and road-context
> signals, sustains driver alertness measurably longer than threshold alarms,
> at a cost and form factor deployable on existing Indonesian fleets.

Everything in §6 exists to test that sentence.

---

## 3. Architecture

v1 described three layers. v2 keeps them and names the thing that was
implicitly sitting between them — the **fusion and decision core**, which is
where the actual intellectual content lives.

```
┌─────────────────────┐   ┌─────────────────────┐
│ L1 Driver State     │   │ L2 Road & Vehicle   │
│ camera → face mesh  │   │ OBD/GPS/dashcam     │
│ PERCLOS, microsleep │   │ speed, lane, light  │
│ head pose, gaze     │   │ following distance  │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           └───────────┬─────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  FUSION CORE                 │
        │  drowsiness_level 0–4        │
        │  + confidence + workload     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  STATE MACHINE               │
        │  driving state · gating      │
        │  cooldowns · escalation      │
        └──────┬────────────────┬──────┘
               │ reflex         │ dialogue
               ▼                ▼
        ┌────────────┐   ┌──────────────────┐
        │ L3a ALERT  │   │ L3b COMPANION    │
        │ pre-rendered│  │ dialogue mgr +   │
        │ <300 ms     │  │ local LLM ≤3 s   │
        └────────────┘   └──────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ efficacy monitor│ ──┐
              └─────────────────┘   │ feeds back into
                       ▲────────────┘ escalation policy
```

**Design invariants** — these are not preferences, they are rules the code must
enforce and the tests must check:

- **I1.** No LLM output ever gates an emergency response. Reflex alerts are
  pre-rendered audio files selected by ID.
- **I2.** The state machine is pure — no I/O, no clock reads, no model calls.
  Time and sensor data arrive as arguments.
- **I3.** Every intervention is rate-limited and logged with the telemetry
  snapshot that triggered it. No unlogged actions.
- **I4.** The system degrades, never crashes. A dead sensor lowers confidence
  and narrows behaviour; it does not stop the loop (§5.4).
- **I5.** Raw video never leaves the device and is never written to disk outside
  an explicit, opt-in research recording session (§8).

---

## 4. Layer specifications

### 4.1 Layer 1 — Driver state monitoring

**Input:** RGB camera (day) / NoIR camera + IR illuminator (night), 640×480, 15–30 FPS.

**Pipeline:** frame → face detection → 478-point face mesh → per-frame features
→ sliding-window aggregation → drowsiness estimate.

**Per-frame features**

| Feature | Definition | Purpose |
|---|---|---|
| EAR | Eye Aspect Ratio, both eyes, from 6 landmarks each | eye closure |
| MAR | Mouth Aspect Ratio | yawning |
| Head pose | pitch / yaw / roll via PnP on mesh | head nod, gaze-off-road |
| Iris offset | iris centre vs. eye corners | gaze direction |
| Face present | detection confidence | occlusion / driver absent |

**Windowed features** (60 s sliding window, updated every 1 s)

| Feature | Definition | Why this one |
|---|---|---|
| **PERCLOS** | % of window with eye closure >80% | The validated standard drowsiness metric (Wierwille et al.); everything else is supporting evidence |
| Microsleep count | closure episodes ≥500 ms | Strongest single danger signal |
| Blink duration mean | mean closure episode length | Rises before PERCLOS does |
| Yawn rate | MAR peaks per minute | Early, noisy indicator |
| Head-nod events | pitch excursions >15° returning to baseline | Late-stage indicator |
| Gaze-off-road ratio | % of window gaze outside road region | Distraction, not drowsiness — kept separate |

**Output — the 0–4 level.** v1 never defined this. Proposed mapping, with the
thresholds themselves treated as tunable parameters to be *fitted* on the
datasets in §6.4, not asserted:

| Level | Name | Trigger condition (initial) |
|---|---|---|
| 0 | Alert | PERCLOS < 8% |
| 1 | Mild | PERCLOS 8–15%, or yawn rate ≥ 2/min |
| 2 | Moderate | PERCLOS 15–25%, or blink duration > 400 ms |
| 3 | Severe | PERCLOS > 25%, or ≥1 microsleep in window, or ≥2 head-nods |
| 4 | Critical | ≥2 microsleeps in window, or a single closure ≥2 s |

**Closure is measured per driver, not against a constant.** Baseline eye
aperture varies two- to threefold across people, so a universal `EAR < 0.20` is
at once too twitchy for one driver and too deaf for the next. Two reference
points are learned per driver while they are known to be alert — open and
closed — and every later frame is scored on that personal scale:

    openness = (ear - closed_ref) / (open_ref - closed_ref)

PERCLOS P80 then becomes literally `openness < 0.20`, the same number for every
face. This needs no model and no training data (`awas/driver/calibration.py`,
`python -m scripts.demo_calibration`) and it removes the largest single error
source in threshold detection before T1 is even trained.

**Hysteresis is mandatory.** Escalation requires the condition to hold for 3
consecutive updates; de-escalation requires 10. Without this the level chatters
across a threshold and the driver is alerted, un-alerted, and re-alerted within
seconds. This is the most common failure mode in naive DMS implementations.

**Distraction** (phone use, gaze-off-road) is a **separate output channel**, not
a drowsiness level. They have different dynamics and demand different responses:
distraction wants an immediate short correction, drowsiness wants sustained
engagement. v1 conflated them; the prototype keeps `phone_usage` separate, which
is right.

**Model choice:** MediaPipe Face Landmarker for the mesh. Its per-frame cost on
a Pi 5 is **unmeasured** — budget the 50 ms of §7.2 for mesh-plus-features and
confirm on the device rather than trusting an estimate. Phone-in-hand detection via a small YOLO-n classifier on the upper-body
crop, only if the gaze/pose heuristic proves insufficient — measure first.

### 4.2 Layer 2 — Road & vehicle context

**Scoped deliberately narrow.** Build only signals that change an intervention
decision. Everything else is cut.

**In scope**

| Signal | Source | Decision it changes |
|---|---|---|
| Speed | OBD-II, GPS fallback | Whether the vehicle is moving at all; risk scaling |
| Vehicle stopped duration | derived | Traffic-jam vs. light vs. parked state |
| Lane position + SDLP | forward camera, lane detection | **Drowsiness evidence** (see §4.3) — the key addition |
| Following distance | forward camera, monocular estimate | Workload; "traffic is moving" nudge |
| Traffic light state | forward camera, colour + shape | Green-light-while-drowsy override |
| Road class + speed limit | GPS + offline map | Workload estimate (highway vs. urban) |

**Explicitly out of scope for this project** (and we say so, rather than
over-claiming): pedestrian detection, general obstacle detection, collision
prediction, sign recognition beyond traffic lights. These are ADAS functions.
AWAS is not an ADAS replacement — v1 says this itself, and we now hold the line
on it in the engineering.

**Workload estimate.** A scalar 0–1 derived from speed, road class, following
distance, and steering activity. High workload suppresses conversation (§5.2).
This is what makes AWAS *not* a distraction.

### 4.3 The fusion core — the research contribution

Facial drowsiness detection alone has two well-known problems: it fails on
occlusion (sunglasses, glare, mask, hand on face) and it detects drowsiness
only once it is visible on the face — which is late.

**Vehicle behaviour degrades earlier than facial signals.** Lane-position
variability (SDLP), steering reversal rate, and steering entropy are validated
drowsiness correlates in the driving-research literature. Fusing them with
facial features gives:

- **Earlier onset detection** (H2 in §6.2) — the falsifiable claim.
- **Graceful behaviour under occlusion** — vehicle signals carry the estimate
  when the face is unavailable.
- **Fewer false positives** — a driver who looks drowsy but is driving with
  tight lane control is probably fine; agreement between channels raises
  confidence, disagreement lowers it.

**Fusion method:** start with a transparent weighted-evidence model (facial
evidence, vehicle evidence, each with a confidence term) rather than a learned
black box. Reasons: it is explainable to judges, it is debuggable, it needs no
paired training data we do not have, and it gives an honest baseline for a
learned model later. Log everything so a learned fusion can be fitted from
collected data as a stretch goal.

**Output of the core:**
```
drowsiness_level : int 0–4
confidence       : float 0–1     # how much to trust the level
workload         : float 0–1     # how busy the driving task is right now
distraction      : enum          # none | gaze_off | phone | passenger
```

### 4.4 Layer 3 — Intelligent companion

Split into two paths (invariant I1).

**L3a — Reflex alerts.** Pre-rendered audio, selected by ID, played through a
pre-warmed audio device. Budget: <300 ms from decision to first sound. Used for
level 4, phone-in-hand, and green-light-while-unresponsive. Never generated.

**L3b — Companion dialogue.** Short, adaptive, contextual conversation.
Hybrid design rather than pure LLM:

1. **Dialogue manager** picks an *intent* (check-in, engage, escalate, confirm-alert)
   from state + level + history.
2. **Template bank** provides a grounded opener for that intent. Always
   available, ~0 ms, always safe.
3. **Local LLM** varies the phrasing and handles the driver's free-form reply.
   Constrained: ≤25 tokens out, ≤3 s budget, safety-filtered.
4. **Fallback:** if the LLM misses its budget or the safety filter rejects the
   output, the template ships instead. The driver never hears silence.

This is a real improvement on "put an LLM on a Pi." It gets conversational
variety where variety helps (habituation resistance) without putting an
unbounded, unverified generator between a drowsy driver and a warning.

**Speech in:** small offline STT (whisper.cpp tiny / Vosk) for driver replies —
and note the reply itself is data: a driver who does not answer within 5 s is
*more* concerning than one who answers. Response latency becomes a drowsiness
signal in its own right.

**Speech out:** Piper TTS, offline, Indonesian voice, pre-warmed.

---

## 5. The intervention controller

### 5.1 Escalation ladder

| Level | Response | Path | Can be suppressed by workload? |
|---|---|---|---|
| 0 | nothing | — | — |
| 1 | Ambient: light conversational check-in | L3b | Yes |
| 2 | Engage: active question requiring a verbal reply | L3b | Yes, briefly |
| 3 | Insist: firm prompt + suggest rest stop, repeat if no reply | L3b + L3a | No, only deferred |
| 4 | Critical: loud reflex alert, sustained, urge immediate stop | L3a | No |

### 5.2 Context gating

An intervention fires only if **all** hold:
- cooldown for its level has elapsed (30 s conversational, 5 s reflex),
- `workload` is below the level's ceiling (level 1 needs workload <0.4; level 4 ignores workload entirely),
- driving state permits it (no conversation while PARKED or SHUTDOWN),
- `confidence` exceeds the level's floor — a low-confidence level 3 is treated
  as a level 2 rather than acted on at full strength.

This is the "knows when to shut up" property. It is also directly testable
(H3, §6.2).

### 5.3 Efficacy feedback — the closed loop

After every intervention, record:
- did the driver respond verbally, and how fast,
- `drowsiness_level` trajectory over the following 5 minutes,
- **time-to-relapse**: seconds until the level returns to the trigger value.

Two uses. **Online:** an intervention type that is not working for this driver
on this trip gets escalated faster next time; one that works gets reused.
**Offline:** time-to-relapse is the primary outcome measure for H1 (§6.3). The
same instrument that makes the product adaptive makes the research measurable —
which is why it is worth building properly rather than bolting on later.

### 5.4 Degradation ladder (invariant I4)

| Failure | Detection | Behaviour |
|---|---|---|
| Camera dark / face lost >10 s | face-present confidence | Drop to vehicle-only estimate, cap level at 3, announce once |
| OBD-II absent | no response on init | GPS speed; disable parking-brake and RPM logic |
| GPS lost | no fix >30 s | Dead-reckon from last speed for 60 s, then drop road context |
| LLM over budget | 3 s timer | Ship the template |
| TTS failure | audio device error | Fall back to reflex tones |
| Thermal throttle | Pi temp >80 °C | Drop camera to 10 FPS, disable LLM, templates only |
| Total sensor loss | all channels stale | Announce "AWAS is not monitoring", log, keep loop alive |

Every row here is a test case.

---

## 6. Evaluation protocol

**This is the section that was missing from v1, and it is what the competition
is judged on.**

### 6.1 Research questions

- **RQ1.** Does adaptive conversational intervention sustain alertness longer
  than a conventional threshold alarm?
- **RQ2.** Does fusing facial and vehicle-behaviour signals detect drowsiness
  onset earlier and with fewer false alarms than facial signals alone?
- **RQ3.** Does context gating reduce intervention burden without increasing
  missed critical events?

### 6.2 Hypotheses (falsifiable, with direction)

- **H1.** Median time-to-relapse after a conversational intervention exceeds
  that after a tone alarm.
- **H2.** Fused detection reduces detection latency at matched false-alarm rate,
  versus facial-only detection.
- **H3.** Context gating reduces total interventions per hour without increasing
  missed level-4 events.

Each hypothesis has a null we are willing to report. If H1 fails, that is a
publishable result and we say so — a system that talks to drivers and *doesn't*
help is worth knowing about.

### 6.3 Conditions

| ID | Condition | Purpose |
|---|---|---|
| A | No system | Floor |
| B | PERCLOS threshold + tone alarm | Conventional ADAS baseline — **implemented fairly**, not as a straw man |
| C | AWAS, full | The claim |
| D | AWAS, context gating disabled | Ablation for H3 |
| E | AWAS, facial-only detection | Ablation for H2 |

### 6.4 Data and environments

**Public datasets** (offline detection evaluation, H2):
NTHU-DDD, UTA-RLDD, DROZY, YawDD. *Action item: several require an application
and a signed agreement — start these early, they take weeks.*

**Driving simulator** (intervention evaluation, H1/H3): a simulator with
telemetry export for lane position and steering. ETS2 (already noted in the v1
prototype README) has a telemetry SDK and is a workable choice; a research sim
would be better if accessible. Participants drive a monotonous route under mild
sleep restriction, within ethics limits.

**Parked vehicle** (hardware validation): thermal, power, camera performance in
real cabin lighting, day and night, no vehicle in motion.

**On-road** (deployment validation only): normal alert driving, to measure false
alarms in real conditions. **No on-road drowsy-driving experiments, ever.**

### 6.5 Metrics

**Detection**
- Precision / recall / F1 on the drowsy class
- **Detection latency**: seconds from ground-truth onset to system level ≥3
- **False alarms per hour** — the metric fleets actually care about
- Performance under occlusion (glasses, sunglasses, low light) reported separately

**Intervention**
- **Time-to-relapse** (primary outcome, H1)
- PERCLOS area-under-curve over the 5 min following an intervention
- Response rate and response latency to prompts
- KSS (Karolinska Sleepiness Scale) self-report at fixed intervals

**Driving performance** (simulator)
- **SDLP** — standard deviation of lane position, the gold-standard drowsy-driving measure
- Lane departures per minute
- Reaction time to lead-vehicle braking events

**System**
- End-to-end latency p50 / p95, per path (reflex vs. dialogue)
- Sustained FPS, CPU load, SoC temperature over a 2 h run
- Power draw; behaviour on power loss

### 6.6 Analysis

Within-subject design where possible (each participant experiences multiple
conditions, order counterbalanced) — it needs far fewer participants than
between-subject, which matters at our scale. Report effect sizes and confidence
intervals, not just p-values. Pre-register the analysis plan before collecting
data so the result is credible either way.

**Be honest about n.** A high-school project will have a small participant pool.
Say so, report it as a limitation, and let the effect size carry the argument.
Judges respect a well-run small study far more than an overstated one.

---

## 7. Hardware

### 7.1 Revised platform

| Item | v1 BOM | v2 | Reason |
|---|---|---|---|
| Compute | Pi 4B | **Pi 5, 8 GB** | CV + LLM concurrently; 4B cannot |
| Cooling | none | **Active cooler — mandatory** | Sustained inference throttles a bare Pi within minutes; a throttled safety system is a broken one |
| Storage | SD card | **NVMe HAT + SSD** | Model load time, log write throughput, SD wear-out under continuous logging |
| Accelerator | none | **AI HAT+ (Hailo-8L), conditional** | Only if profiling shows road CV cannot share the CPU. Decide with measurements, not in advance |
| IR illuminator | **missing** | **added** | See §7.3 |
| Backup power | 2× battery | **UPS HAT / supercap** | Purpose is clean shutdown on ignition cut, not runtime |

### 7.2 Latency budget

| Stage | Budget | Notes |
|---|---|---|
| Capture → frame available | 30 ms | |
| Face mesh + features | 50 ms | MediaPipe, CPU |
| Window update + fusion | 5 ms | |
| State machine | <1 ms | pure logic |
| **Reflex alert: decision → audible** | **<300 ms** | **measured 25 ms** on an M1 Pro — pre-warmed stream, clips already in memory, stamped in the device callback (`awas/speech/audio.py`) |
| Dialogue: decision → speech starts | <3000 ms | template fallback at 3 s |
| **Total perception loop** | **<100 ms (≥10 Hz)** | |

Thermal budget: sustained SoC <75 °C at 25 °C ambient. Cabin ambient in
Indonesia can exceed 40 °C — test at 40 °C ambient, not room temperature.

### 7.3 Two defects in the original BOM

1. **NoIR camera with no IR illuminator.** A NoIR module removes the IR-cut
   filter so it *can* see infrared — but nothing in the BOM emits any. At night
   the cabin is dark and the camera sees nothing. Needs an 850 nm IR LED array,
   eye-safe at cabin distance, ~Rp150k. Without this, night operation — half
   the use case, and the half where drowsiness matters most — does not work.
2. **Backup battery sized as runtime power.** For a device that must survive
   ignition cut, what is needed is a clean-shutdown supply (UPS HAT or
   supercapacitor), not two general-purpose cells.

Also worth revisiting: the 3.5" touchscreen. Its purpose should be stated —
if it exists for setup and status only, a cheaper non-touch display or a
phone-based setup flow may serve better. If it displays anything while driving,
that needs its own distraction justification.

### 7.4 Making "vehicle-agnostic" honest

v1 claims applicability from private cars to buses, trucks, and rail. The
software is agnostic; **the OBD-II assumption is not**:

- Heavy commercial vehicles use **SAE J1939**, not OBD-II.
- Many older Indonesian vehicles expose speed unreliably or not at all.
- Rail has no equivalent interface.

Therefore: **GPS speed is the primary vehicle-signal path, OBD-II is an
enhancement.** Any signal only OBD-II can provide (parking brake, RPM) must be
optional, with the system fully functional without it. This costs little and
makes the national-scale claim actually true.

---

## 8. Privacy, ethics, and research safety

**Privacy policy — concrete, not aspirational.**
- Raw video never leaves the device. No cloud, no network path for frames.
- Raw video is never written to disk except during an explicit, opt-in,
  time-boxed research recording session with written consent.
- Normal operation persists only derived numeric features and event logs.
- Logs are local and driver-clearable.
- A hardware indicator shows when the camera is active.
- For fleet deployment, what the operator can and cannot see is a policy
  decision that must be written down before deployment, not after.

**Research ethics.**
- No experiment ever places a drowsy person on a public road. Simulator or
  parked vehicle only.
- Informed consent, including the right to stop at any point, no explanation needed.
- Sleep restriction, if used, stays within documented safe limits, with
  participants not driving themselves home.
- Face data from participants is stored locally, pseudonymised, and deleted on request.

**Safety framing.** AWAS assists an attentive driver. It is not an autonomous
system, not a substitute for rest, and not an ADAS. The system says this to the
driver on first run, and the project says it in every document. Overclaiming a
safety system is both an ethical failure and, for judges, an obvious one.

---

## 9. Roadmap

### Milestone 1 — Laptop end-to-end demo *(current target)*

Everything running on the MacBook with a webcam. No hardware purchase required.
Proves the full loop and produces the demo video.

- [x] State machine in the core: pure, exhaustively tested
- [ ] Layer 1 on webcam: face mesh → EAR/MAR/head pose → PERCLOS window
- [ ] Fusion core producing level + confidence, with hysteresis
- [ ] Reflex alert path with pre-rendered audio, latency-measured
- [ ] Dialogue manager + template bank (no LLM yet)
- [ ] TTS output (system TTS on macOS, Piper later)
- [ ] Live debug overlay: video + landmarks + level + state + action log
- [ ] Structured session logging in the format §6 will analyse

**Exit criterion:** sit in front of the laptop, let your eyes close, and the
system talks to you — correct level, correct timing, no chatter, logged.

### Milestone 2 — Detection quality
Public datasets acquired and evaluated. Thresholds fitted rather than guessed.
Facial-only baseline measured (condition E). Occlusion cases characterised.

### Milestone 3 — Simulator and vehicle signals
Sim telemetry integration, SDLP and steering features, fusion of both channels.
Conditions A–E runnable end to end. **H2 testable here.**

### Milestone 4 — Companion
Local LLM integrated behind the template fallback, budget-enforced, safety
-filtered. STT for driver replies. Response latency as a signal. Efficacy loop closed.

### Milestone 5 — Pi 5 deployment
Port, profile, tune. Latency, thermal, and power budgets verified at 40 °C
ambient. Parked-vehicle validation, day and night.

### Milestone 6 — Study and write-up
Run the protocol. Analyse per the pre-registered plan. Report what happened,
including whatever failed.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset access denied or slow | H2 unevaluable | Apply immediately; collect a small own dataset in parallel as backup |
| Pi 5 cannot sustain CV + LLM | Milestone 5 slips | Template-only mode is a complete, shippable system; LLM is an enhancement, not a dependency |
| Too few participants for significance | Weak H1 result | Within-subject design; report effect sizes; be explicit about n as a limitation |
| Simulator telemetry insufficient for SDLP | H1/H3 weakened | Verify telemetry export **before** building on it — this is a Milestone-3 blocker, check it in Milestone 1 |
| Night performance poor | Half the use case lost | Buy the IR illuminator early; test night before committing to the enclosure design |
| Scope creep back into full ADAS | Nothing finishes | §4.2's out-of-scope list is binding. Revisit only after Milestone 4 |
| Conversation itself distracts | Thesis inverted | This is exactly what H3 and the SDLP measurements test. If it distracts, we will see it and report it |

---

## Appendix — open questions to settle

1. Which simulator? Verify telemetry export for lane position and steering **before** Milestone 3.
2. Indonesian or English dialogue for the demo? Piper has `id_ID` voices; the LLM's Indonesian quality at 1–2 B parameters needs testing.
3. Is the touchscreen justified? What does it show, and when?
4. Fleet-operator visibility policy — needed before any pilot conversation.
5. Does the competition require the submission in Indonesian? This plan is in
   English to match the codebase; a translated submission version may be needed.
