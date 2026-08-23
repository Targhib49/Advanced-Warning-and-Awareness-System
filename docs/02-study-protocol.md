# Study Protocol

**Testing H1: does conversational intervention sustain alertness longer than a
tone alarm?**

Operational companion to [`00-master-plan.md`](00-master-plan.md) §6. That
section says what we intend to measure; this one says exactly what happens in
the room, in what order, with what safeguards, and how to run the analysis
afterwards.

Read this end to end before recruiting anyone.

---

## 1. What is actually being tested

We already have strong evidence the system **detects** drowsiness: 98.6% on
seven subjects the model never saw, against a logistic-regression baseline of
86.8%. That result is finished and does not need participants.

What we have no evidence for is the thesis itself — that *talking to a drowsy
person keeps them alert longer than beeping at them does*. Everything below
exists to produce that evidence, or to fail to and say so.

**H1.** Median time-to-relapse after a conversational intervention (condition C)
exceeds that after a tone alarm (condition B).

**Null we are willing to report.** No difference, or conversation performing
worse. If that is the result, it goes in the paper. A system that talks to
drowsy people and does not help them is worth knowing about, and reporting it
honestly is better science than a study designed so it cannot fail.

---

## 2. Safety and ethics — read this first

**No one drives.** Not on a road, not in a moving vehicle, not once. Every
session happens at a desk or in a stationary vehicle with the engine off. The
system is fed a live camera; there is no vehicle under anyone's control.

**No induced sleep deprivation.** We do not ask anyone to stay up, skip sleep,
or arrive tired. We schedule sessions at times when natural alertness is
already low — late evening, or early afternoon — and work with the sleepiness
people bring. This costs us effect size and it is not negotiable: deliberately
impairing a minor for a school project is not something to be clever about.

**Participants under 18 need a parent or guardian to consent**, in writing, in
addition to the participant's own assent. Both forms are in §7.

**Anyone may stop at any moment**, with no explanation and no consequence.
Say this out loud at the start of every session, not just on the form.

**Exclusions.** Do not enrol anyone who: has a diagnosed sleep disorder, has
photosensitive epilepsy or any condition triggered by sudden sounds, is taking
medication affecting alertness, or is under 13.

**If a participant becomes distressed or unusually impaired**, stop the
session, stay with them, and make sure they get home safely and not by driving.

**Data.** Sessions produce derived numbers only — no video is recorded unless
the participant separately opts in, and no session file leaves the machine that
made it. Pseudonymise on collection: participant `P03`, never a name. Delete on
request without asking why.

---

## 3. Design

**Within-subject, counterbalanced.** Every participant experiences both
conditions. This is the only design that gives a usable answer at the sample
size a school team can recruit — between-subject would need an order of
magnitude more people to see anything.

| | |
|---|---|
| Conditions | **B** tone alarm · **C** AWAS conversation |
| Order | Counterbalanced: half do B→C, half C→B |
| Session length | 40 minutes per condition |
| Gap between conditions | ≥ 24 hours, same time of day |
| Target n | 8–12 participants (16–24 sessions) |

**Why counterbalance.** People get sleepier as an evening goes on. If everyone
did B first and C second, C would look better purely because it ran later.
Alternating the order cancels that out. Assign by arrival: P01 → B first,
P02 → C first, P03 → B first, and so on.

**Condition A** (no intervention) is available in the code and is the honest
floor, but it means watching someone get drowsy and doing nothing, for forty
minutes. Run it only if you have a participant who volunteers specifically for
it and you are comfortable stopping the moment they reach level 4.

---

## 4. Session runbook

Print this. Follow it exactly the same way every time — an inconsistent
protocol is the fastest way to a result that means nothing.

### Before the participant arrives

- [ ] Camera positioned as it would be in a cabin: slightly below eye level, driver's face filling roughly a third of the frame
- [ ] Room lighting fixed and noted — the same for every session
- [ ] `python -m scripts.render_phrases` has been run at least once
- [ ] Test the audio at the volume you will actually use
- [ ] Know the condition and order for this participant before they walk in

### Arrival (10 minutes)

- [ ] Explain what the system does, in plain language
- [ ] Give the information sheet (§7), let them read it without hurrying
- [ ] Consent form signed — plus guardian consent if under 18
- [ ] Say out loud: *"You can stop at any point, for any reason, and you do not have to tell me why."*
- [ ] Record the pseudonym, condition, order, date and start time on paper

### Baseline (5 minutes)

- [ ] Participant sits normally, faces forward, no phone
- [ ] Start the session — calibration runs automatically
- [ ] Wait for the overlay to show **`personal scale`**. Do not proceed until it does; without it the system is running on population defaults and the session is not measuring what you think it is
- [ ] Ask for a **KSS rating (1–9)** and press that number key to log it

### Main block (40 minutes)

- [ ] Participant performs a low-stimulation task — watching a monotonous video, or a long dull reading passage. No conversation from you
- [ ] Ask for a **KSS rating every 10 minutes**, log each with the number key
- [ ] Let the system intervene as its condition permits. Do not prompt, coach, or react
- [ ] Note anything unusual on paper with the clock time — a laugh, a cough, someone walking in

### Close (5 minutes)

- [ ] Final **KSS rating**, logged
- [ ] Press `q` to end the session cleanly so the log closes and the profile saves
- [ ] Ask two open questions and write the answers verbatim:
      *"What did you notice the system doing?"* and *"Did it help, annoy you, or neither?"*
- [ ] Confirm they are fine, and that they are not driving home
- [ ] Confirm the session file exists in `data/sessions/`

### Commands

```bash
# Condition B — tone alarm baseline
python -m scripts.run_webcam --driver P03 --condition B

# Condition C — full AWAS
python -m scripts.run_webcam --driver P03 --condition C
```

The condition is written into the session log, so the analysis groups by it
automatically. Getting the flag wrong silently mislabels a whole session —
check it before every run.

---

## 5. Measures

**Primary — time-to-relapse.** Seconds from an intervention until drowsiness
returns to the level that triggered it. Computed from the session log by
`scripts/analyze_session.py`. Longer is better.

**Secondary**

| Measure | Source |
|---|---|
| KSS trajectory across the session | number-key annotations |
| PERCLOS area under curve, 5 min post-intervention | session samples |
| Interventions per hour | action records |
| Peak level reached | session samples |
| Reported annoyance | the closing questions |

**Two properties of time-to-relapse that must not be got wrong**

*Recovery is required before a relapse counts.* If the participant never drops
below the trigger level, that is one continuous episode, not a relapse.

*Recovered-but-never-relapsed is censored, not zero.* An intervention so
effective the participant stayed alert for the rest of the session is the best
possible outcome. Recording it as zero would invert the result. The analysis
script counts these separately — report the count alongside the medians.

---

## 6. Analysis

Decide all of this **before collecting any data** and write it down. Choosing
the analysis after seeing the numbers is how honest people accidentally produce
dishonest results.

```bash
python -m scripts.analyze_session --compare data/sessions/*.jsonl
```

**Pre-registered plan**

1. Primary comparison: median time-to-relapse, condition C versus condition B, paired within participant.
2. Report the **effect size and its confidence interval**, not a p-value alone. At n≈10 a p-value tells you very little; the size of the difference tells you something.
3. Report censored observations as counts per condition. Never impute them.
4. Report every participant, including anyone who withdrew and at what point.
5. Report KSS separately as a subjective check on whether the objective measure agrees with how people felt.

**State the limitations plainly in the write-up**

- Small sample, one site, participants who know each other and know what is being tested
- Natural sleepiness rather than controlled sleep restriction, so the range of drowsiness is narrow
- Desk-based, not driving: no steering, no risk, no real consequence for failing to stay alert
- Participants know which condition is which, and so do the experimenters

None of these invalidate the study. Concealing them would.

---

## 7. Participant materials

### 7.1 Information sheet

> **What this study is about**
>
> We have built a system that watches for signs of drowsiness using a camera,
> and responds when it sees them. We want to find out whether a system that
> *talks* to you helps you stay alert longer than one that just beeps.
>
> **What you will do**
>
> You will sit at a desk and do a deliberately boring task for about 40
> minutes, twice, on two different days. A camera will watch your face. Every
> ten minutes we will ask how awake you feel on a scale of 1 to 9. Sometimes
> the system will make a sound or say something.
>
> **You will not drive, and you will not be asked to lose any sleep.** We will
> schedule sessions at a time when people are naturally a bit tired.
>
> **What we record**
>
> The camera works out numbers about your eyes — how open they are, how often
> you blink. **We do not record or save any video of you.** What gets saved is
> those numbers, what the system did, and your 1-to-9 ratings, under a code
> like "P03" rather than your name.
>
> **You can stop at any time**, for any reason, without explaining. Nothing
> happens if you do. You can also ask us to delete your data afterwards and we
> will, no questions.
>
> **Risks.** The system sometimes makes a sudden sound. If sudden sounds are a
> problem for you, please do not take part.
>
> Questions at any point, before, during or after: [YOUR NAME AND CONTACT]

### 7.2 Consent form

> Please initial each line:
>
> ☐ I have read the information sheet and had a chance to ask questions.
> ☐ I understand I can stop at any time without giving a reason.
> ☐ I understand no video of me will be recorded or saved.
> ☐ I understand I can ask for my data to be deleted afterwards.
> ☐ I agree to take part.
>
> Participant name: ......................................
> Signature: .............................. Date: ..........
>
> **For participants under 18 — parent or guardian:**
>
> ☐ I have read the information sheet and I consent to my child taking part.
>
> Name: .................................. Relationship: ..........
> Signature: .............................. Date: ..........

### 7.3 Session record sheet

Keep this on paper, separately from the session files.

```
Participant code   P___        Date ____________
Condition          B / C       Session 1st / 2nd
Time started       ______      Time ended  ______
Room lighting      ______________________________
Calibration reached "personal scale"?   yes / no

KSS   0min ___  10min ___  20min ___  30min ___  40min ___  end ___

Notes / interruptions
_________________________________________________________________
_________________________________________________________________

"What did you notice the system doing?"
_________________________________________________________________

"Did it help, annoy you, or neither?"
_________________________________________________________________
```

---

## 8. Realistic scope for the remaining weeks

You will not complete a full study before the competition, and you should not
pretend otherwise. What is achievable:

| | |
|---|---|
| **Achievable now** | 3–4 participants, both conditions each, ~6–8 sessions. Enough to demonstrate the protocol runs, the instrument works, and to show a preliminary direction |
| **Present it as** | "a pilot study of n=4, with the full protocol pre-registered" — which is an honest and respectable thing to show judges |
| **Do not present it as** | proof that H1 holds |

A pilot that is small and honest beats a bigger study that overstates. Judges
have seen overstatement before; they have seen far less of a team that knows
exactly what its own evidence does and does not support.

The one thing worth more than extra participants: **run the pilot at all.**
Having a real, if small, result about intervention efficacy puts you in a
different category from every project that only demonstrates detection.
