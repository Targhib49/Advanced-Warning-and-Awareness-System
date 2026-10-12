# IV. RESULTS AND DISCUSSION

The contribution evaluated in this section is the **road-awareness layer**: a
vehicle-behaviour channel derived from lane position, fused with conventional
facial monitoring. Facial drowsiness detection is treated here as the
established baseline against which that channel is measured, not as a result in
itself.

Three properties are examined, each isolating something the facial channel
cannot do alone: whether the road channel detects onset earlier (§D-1), whether
it continues working when the driver's face is unavailable (§D-2), and whether
disagreement between channels suppresses facial false positives (§D-3).

Results measured on people are distinguished throughout from results measured
in simulation. Where simulation is used, the reason is given.

---

## A. Road Awareness: Lane Position Estimation

Lane position is recovered by perspective warp, adaptive thresholding and a
sliding-window trace, with metres-per-pixel derived per frame from the detected
lane width. No learned component is involved. Lane markings are high-contrast
painted lines of known geometry, and a classical pipeline resolves them in
approximately 3 ms — against 38 ms for a comparable learned detector, which
matters directly for the embedded target.

Accuracy was verified by rendering lanes at known offsets in bird's-eye space,
inverse-warping them to the camera view, and recovering the offset. The ground
truth is therefore exact by construction.

**TABLE I. LANE OFFSET RECOVERY**

| True offset (m) | Detected (m) | Error (m) |
|---:|---:|---:|
| −0.80 | −0.783 | 0.017 |
| −0.40 | −0.388 | 0.012 |
| 0.00 | +0.004 | 0.004 |
| +0.40 | +0.420 | 0.020 |
| +0.80 | +0.815 | 0.015 |

Maximum error is 0.020 m over a ±0.8 m range, at 2.98 ms per frame.

**Calibration is not optional, and its failure is silent.** Applied to footage
from a camera mounted differently from the default trapezoid, detection fell to
**0 % of frames while reporting no error**. An automatic procedure —
vanishing-point estimation followed by a scored search over trapezoid geometry —
recovered the mounting (top edge 0.70 against a true 0.70; base half-width 0.62
against 0.62) and restored detection to 100 %. The scoring function requires no
ground truth: a calibration is judged by how often it finds a lane, how
consistent the lane width is, and how stable the offset is between frames.

For deployment the relevant risk is therefore not gradual degradation but
complete, unannounced failure, and the calibration procedure exists to remove it.

---

## B. The Vehicle Channel as Drowsiness Evidence

Standard deviation of lane position (SDLP) over a 60-second window separates
simulated steady driving (0.048 m) from wandering (0.343 m) by a factor of
seven.

Absolute SDLP thresholds are, however, the same mistake as a fixed EAR
threshold. Lane-keeping precision is personal: a driver whose alert SDLP is
0.12 m and one whose alert SDLP is 0.20 m are both driving competently. The
channel therefore learns each driver's alert-driving baseline and scores the
**excess** over it, in metres, rather than comparing against a constant.

**TABLE II. VEHICLE CHANNEL ON A PERSONAL SCALE**

| SDLP excess over baseline | Level | Tidy driver (base 0.12 m) | Looser driver (base 0.20 m) |
|---:|---:|---:|---:|
| +0.00 m | 0.0 | 0.12 m | 0.20 m |
| +0.04 m | 1.3 | 0.16 m | 0.24 m |
| +0.08 m | 2.4 | 0.20 m | 0.28 m |
| +0.12 m | 3.2 | 0.24 m | 0.32 m |

Equal impairment produces an equal level irrespective of baseline. The
increments are expressed in metres because the driving-fatigue literature
reports impairment as absolute SDLP increases, with roughly +0.05 m
corresponding to legally significant alcohol impairment — which makes it a
defensible position for level 2.

**A ceiling found during this analysis.** The thresholds initially used
(level 3 at 0.35 m) placed the top of the scale above the range real wander
occupies, so the channel saturated near level 2.7 and **could not raise a
level-3 alarm under any configuration**. It was not detectable from the design;
it emerged only from sweeping the parameter. The thresholds were lowered to sit
inside the observed range, and the personal scale made primary.

---

## C. Supporting Result: Eye-State Classification

The facial channel provides the baseline the road channel is measured against.
It was trained on the MRL Eye Dataset (84,898 infrared images, 37 subjects) and
partitioned **by subject**, so no participant appears in both training and
evaluation. MRL contains hundreds of near-identical frames per person; a random
per-image split places the same eyes on both sides and inflates accuracy without
measuring generalisation to an unseen driver.

Evaluation used seven held-out subjects (12,246 images).

**TABLE III. EYE-STATE CLASSIFICATION, SUBJECT-WISE PARTITION**

| Method | Parameters | Accuracy (%) |
|---|---:|---:|
| Majority class | — | 54.69 |
| Logistic regression on pixels | 257 | 86.82 |
| EyeStateNet | 23,538 | **98.59** |

**TABLE IV. ACCURACY BY CAPTURE CONDITION**

| Condition | n | Accuracy (%) |
|---|---:|---:|
| No glasses | 10,925 | 99.02 |
| **Glasses** | **1,321** | **95.00** |
| Good lighting | 3,735 | 98.66 |
| Poor lighting | 8,511 | 98.55 |
| RealSense sensor | 10,508 | 98.72 |
| IDS sensor | 1,738 | 97.76 |

Accuracy is insensitive to illumination — 98.55 % against 98.66 % — which
matters because drowsiness concentrates in night driving. It falls four points
with eyewear, and that stratum is reported rather than aggregated away: an
accuracy figure quoted without its worst case is not a useful number, and the
occlusion result in §D-2 is precisely the response to it.

---

## D. Multimodal Fusion: What Road Awareness Adds

Detection latency requires ground truth for the instant impairment began, which
no observational recording provides. The following therefore use **simulated
episodes** (ten runs, 600 s each, onset at 300 s) in which onset is known by
construction. They characterise what the fusion rule does with its inputs, not
accuracy on drivers.

### D-1. Earlier onset detection

The driving-fatigue literature reports that lane keeping degrades before facial
signs become visible. That premise is **assumed** here, not established: the
simulated vehicle channel is given a 45 s lead, and what is measured is how much
of that lead the fusion rule converts into earlier detection.

**TABLE V. ONSET DETECTION LATENCY**

| Configuration | Detection rate | Median latency (s) |
|---|---:|---:|
| Facial only | 1.00 | 58.3 |
| Facial + vehicle | 1.00 | **56.1** |

The improvement is **2.2 s from a 45 s lead** — a conversion of under 5 %. The
cause is the SDLP window: the statistic requires 60 s of lane samples before it
is meaningful, so the channel responds slowly even when the underlying signal
leads substantially. **Window latency consumes most of the available advantage**,
and this is the weakest of the three results. Establishing whether the premise
holds at all for real drivers is hypothesis H2 and requires participants.

### D-2. Robustness to facial occlusion

Here nothing is assumed. Occlusion either removes the facial channel or it does
not, and the outcome is fully determined. The driver's face is made unavailable
from t = 330 s — 30 s after onset, simulating sunglasses, glare or a hand.

**TABLE VI. PERFORMANCE WITH THE FACE OCCLUDED FROM t = 330 s**

| Configuration | Channel available | Onset detected | Median latency (s) |
|---|---:|---:|---:|
| Facial only | 55 % | **0 %** | — |
| Facial + vehicle | 100 % | **100 %** | 98.1 |

This is the strongest evidence for the road-awareness layer. Facial-only
monitoring is blind for 45 % of the episode and **fails to detect the onset in
any of ten runs**. With the vehicle channel present, coverage is continuous and
onset is detected in every run.

The 98.1 s latency is markedly worse than the 58.3 s of an unoccluded facial
channel, and that is the honest characterisation: the road channel is a
**degraded but functioning** fallback, not an equivalent substitute. A system
that detects late is materially better than one that detects never.

### D-3. Suppression of facial false positives

The complementary case: the facial channel misreads an alert driver — the
glasses and glare failure mode of §C — while lane keeping remains steady.

**TABLE VII. FACIAL FALSE POSITIVE, LANE KEEPING STEADY**

| Configuration | Median level | False alarms raised |
|---|---:|---:|
| Facial only | 3.0 | **100 %** |
| Facial + vehicle, asymmetric fusion | 3.0 | 100 % |
| Facial + vehicle, weighted fusion | 1.0 | **0 %** |

Weighted fusion eliminates the false alarm entirely: a driver who appears drowsy
while holding the lane precisely is scored down to level 1. Asymmetric fusion
does not, because by construction it takes the maximum channel level.

Taken with §D-1, this establishes a genuine and unavoidable trade. **A single
fusion operator cannot both detect earlier and suppress disagreement.** The
weighted rule is a mean, so it cannot exceed its most alarmed channel; the
asymmetric rule preserves early warnings but forfeits the suppression. Both are
implemented and selectable, and the choice is a deployment decision about
whether missed detections or false alarms are the more costly failure — which
differs between a private vehicle and a commercial fleet.

### D-4. What the driver is shown

The in-cabin display is a consequence of the results above rather than an
independent design exercise, and it changed materially once the road channel
existed.

**While the vehicle is moving the screen carries almost nothing to read.** The
lit segment and the band colour convey the level at a glance; the words are
audio. Detailed information appears only when the vehicle is stationary. A
display that requires reading is a second distraction competing with the one the
system exists to correct.

Three elements are direct products of the road-awareness layer and did not exist
in the original interface design:

**Channel provenance.** Each screen names the evidence the estimate rests on —
`FACE + LANE`, or `LANE` alone. When the camera is occluded the driver is told
that monitoring continues on lane behaviour, and that the level is capped
because a single channel cannot be fully trusted (§D-2). Without the road
channel this state was simply a failure notice.

**Gate transparency.** When workload suppresses an intervention, the screen says
so rather than falling silent inexplicably. A system that goes quiet in heavy
traffic and does not explain why is indistinguishable from one that has stopped
working, and driver trust is a prerequisite for a system whose whole purpose is
to be listened to.

**Vehicle-channel evidence in the driver's own terms.** The occluded state
reports SDLP as an excess over the driver's own baseline — "+8 cm" — rather than
an absolute figure that means nothing without knowing that driver.

The critical state deliberately breaks the calm: it is the only screen that
floods, and the only one that abandons the glanceable convention, because at
that point the display is no longer supporting a driving task.

---

## E. Real-Time Performance

Measured on an Apple M1 Pro, single-threaded where applicable, with a live
camera and a face in frame.

**TABLE VIII. LATENCY AGAINST DESIGN BUDGET**

| Stage | Measured | Budget | Utilisation |
|---|---:|---:|---:|
| Lane detection (mean) | 2.98 ms | — | — |
| Face mesh, EAR, head pose (p95) | 8.33 ms | 50 ms | 17 % |
| Full perception loop (p95) | 8.34 ms | 100 ms | 8 % |
| Decision → audible alert (worst) | 9.47 ms | 300 ms | 3 % |

The reflex path meets its budget with a 30× margin, achieved structurally rather
than by optimisation: every utterance is rendered to audio before the system
runs, the output stream is opened once at startup, and playing a clip is a
buffer swap. No code path exists from a generator to the loudspeaker.

**A measurement that constrains the architecture.** The object detector
supplying traffic-light state and following distance costs 38 ms per frame,
roughly five times the face mesh. Running it per frame would exceed the
perception budget several times over on the target hardware. It is throttled to
approximately 3 Hz while driver monitoring runs at full rate — admissible
because a traffic light does not change state within 300 ms. **Perception rates
are set by signal dynamics rather than run uniformly**, and the road-awareness
layer is affordable only because of it.

---

## F. Parameter Sensitivity

**TABLE IX. HYSTERESIS DEPTH AND DETECTION LATENCY**

| Escalation ticks | Median latency (s) | False alarms |
|---:|---:|---:|
| 1 | 56.1 | 0 |
| 2 | 57.2 | 0 |
| 3 (deployed) | 58.3 | 0 |
| 5 | 60.4 | 0 |
| 8 | 63.6 | 0 |

Hysteresis costs approximately 1.07 s per tick and eliminates level
oscillation — the failure that trains drivers to disregard alerts.

**TABLE X. WORKLOAD CEILING AND SPEECH OPPORTUNITY**

| Ceiling | Situations permitting speech | Share |
|---:|---:|---:|
| 0.20 | 1 of 7 | 14 % |
| 0.30 | 3 of 7 | 43 % |
| **0.40 (deployed)** | **4 of 7** | **57 %** |
| 0.50 | 5 of 7 | 71 % |
| 0.70 | 6 of 7 | 86 % |
| 1.01 (ungated) | 7 of 7 | 100 % |

The workload estimate is itself a road-awareness product, derived from speed,
speed variability, following distance, road class and recent lane departures. At
the deployed ceiling the system speaks when stationary, on a quiet motorway, on
an arterial and in steady urban traffic, and stays silent in stop-go traffic,
while following closely, and immediately after a lane departure.

**Two sensitivity analyses did not yield reliable orderings and are excluded.**
Sweeps of the PERCLOS thresholds and of the observation-window length produced
detection rates collapsing to 0 or 1 with little intermediate structure, because
outcomes under the synthetic model are dominated by a single rule (§G-1).
Characterising them properly requires recorded footage with graded drowsiness.
They are omitted rather than reported at low confidence.

---

## G. Architectural Findings

Three properties emerged from analysis rather than design, and are the most
transferable result of this work.

**1) Microsleep detection dominates all other facial evidence.** Detection
latency was invariant at 58.3 s across PERCLOS thresholds, window length and
fusion configuration. A single closure exceeding 500 ms raises the level before
any windowed statistic becomes decisive. This is defensible — a microsleep is
unambiguous — but it means tuning PERCLOS cannot improve responsiveness while
that rule stands, and it is a further argument for a second, independent channel.

**2) Confidence-weighted fusion cannot detect earlier than its channels.** A
weighted mean of an alarmed channel and a quiet one lies between them. In an
earlier configuration, adding a vehicle channel leading the facial one by 45 s
made detection 6.4 s *later*. Hypothesis H2 seeks earlier detection through
fusion; disagreement damping seeks fewer false alarms; one averaging operator
cannot deliver both (§D-3).

**3) A channel's dynamic range must reach the alarm level.** The vehicle
channel's SDLP thresholds were set from literature values never fitted to this
pipeline, placing level 3 at 0.35 m while real wander peaked near 0.32 m. The
channel could not raise a critical alarm however early it degraded. Thresholds
transferred from other measurement pipelines require verification that the
resulting scale is reachable — a failure invisible in normal operation.

---

## H. Limitations

**Intervention efficacy is unevaluated.** The claim that conversational
intervention sustains alertness longer than a tone alarm has no supporting data.
Experimental conditions are implemented, sessions are logged in a schema from
which time-to-relapse is computed, and the analysis is pre-registered; the study
has not been run. No result here should be read as evidence for the
conversational intervention.

**The road-awareness results are simulated.** §D characterises the fusion rule's
behaviour given its inputs. §D-1 additionally assumes the literature's premise
that lane keeping degrades first. §D-2 assumes nothing and is correspondingly
the strongest of the three.

**Lane detection is verified geometrically, not on real roads.** Table I
validates the estimator against synthetic lanes of known geometry, not its
robustness to worn markings, shadows, occluding vehicles or curvature. The
polynomial fit tracks near-straight lanes only.

**Detection is evaluated on still images.** MRL provides cropped eye images, so
the temporal components — PERCLOS accumulation, blink dynamics, hysteresis — are
unevaluated against ground truth.

**The SDLP baseline has not been collected from any driver.** The personal scale
in Table II is implemented and tested; no real alert-driving baseline exists yet.

**Calibration transfer across model versions is unverified.** Landmark indices
were confirmed against MediaPipe 0.10.35. The Raspberry Pi target cannot install
that release, and index sets have moved between releases. An incorrect index
produces a plausible but meaningless ratio rather than an error.
