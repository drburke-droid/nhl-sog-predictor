"""Synthesize narration with Piper and build the scene timeline (timeline.json)."""
import json, os, wave, sys
from piper import PiperVoice, SynthesisConfig

HERE = os.path.dirname(os.path.abspath(__file__))
# Piper's en_US-ryan-high voice (https://huggingface.co/rhasspy/piper-voices, MIT); not kept in the repo (120 MB)
VOICE = os.environ.get('PIPER_VOICE', '/home/user/voices/en_US-ryan-high.onnx')
OUT = os.path.join(HERE, 'audio')
os.makedirs(OUT, exist_ok=True)

# (scene id, lead-in seconds, tail seconds, [lines])
# Each line is (spoken text, caption text). Caption None -> same as spoken.
SCENES = [
    ('title', 1.6, 1.2, [
        ("The B-2 Spirit stealth bomber has a wingspan of a hundred and seventy-two feet. That's more than half the length of a football field.",
         "The B-2 Spirit stealth bomber has a wingspan of 172 feet. That's more than half the length of a football field."),
        ("Yet on a radar screen, it's said to look no bigger than a large bird.", None),
        ("So how does something this big disappear?", None),
    ]),
    ('radar', 0.8, 1.0, [
        ("To understand stealth, start with radar.", None),
        ("A radar station sends out a pulse of radio waves.", None),
        ("When that pulse hits an aircraft, some of the energy bounces back as an echo.", None),
        ("By timing how long the echo takes to return, the radar knows exactly how far away the target is.", None),
    ]),
    ('rcs', 0.6, 1.0, [
        ("How strong that echo is depends on the target's radar cross section, or R. C. S.",
         "How strong that echo is depends on the target's radar cross section, or RCS."),
        ("A conventional bomber like the B-52 has a radar cross section of roughly a hundred square meters.", None),
        ("Public estimates put the B-2 at a thousand times smaller, or less. The real number is classified.", None),
        ("It gets there with four main tricks.", None),
    ]),
    ('shape', 0.6, 1.6, [
        ("Trick one: shape.", None),
        ("Flat surfaces and right-angle corners act like mirrors, throwing radar energy straight back to where it came from.", None),
        ("The B-2 has no vertical tail, no boxy fuselage, and almost no right angles.", None),
        ("Its smooth, blended flying-wing shape deflects incoming waves off to the side, away from the radar that sent them.", None),
    ]),
    ('edges', 0.6, 1.4, [
        ("Trick two: edge alignment.", None),
        ("Every edge on an aircraft scatters some energy, strongest in the direction perpendicular to that edge.", None),
        ("So the designers lined up the B-2's edges along just a few angles. Each segment of the sawtooth trailing edge runs parallel to one of the leading edges.",
         "So the designers lined up the B-2's edges along just a few angles. Each segment of the sawtooth trailing edge runs parallel to a leading edge."),
        ("Instead of glowing in every direction, the reflections are squeezed into a few narrow spikes, aimed away from the nose, where threats are most likely to be.", None),
        ("Even the doors and panels get zigzag edges, for the same reason.", None),
    ]),
    ('ram', 0.6, 1.4, [
        ("Trick three: radar-absorbent materials.", None),
        ("Whatever energy the shape can't deflect, the skin tries to soak up.", None),
        ("The B-2's surface is built from composites and special coatings that turn radio waves into a tiny amount of heat.", None),
        ("Some layers are tuned so that waves bouncing off the top and the bottom of the coating cancel each other out, like noise-cancelling headphones for radar.", None),
        ("Even panel seams and fasteners are filled and taped, because a tiny gap can light up on radar.", None),
    ]),
    ('engines', 0.6, 1.4, [
        ("Trick four: hiding the engines.", None),
        ("A jet engine's spinning fan blades are one of the brightest radar reflectors on any aircraft.", None),
        ("On the B-2, all four engines are buried inside the wing, fed by intakes on top, where radars on the ground can't see them.", None),
        ("Curving, S-shaped ducts mean radar can never see straight down to the fan, and waves that go in get absorbed as they bounce off the lined walls.", None),
        ("The exhaust leaves through shallow troughs on top of the wing, hiding the hot gases from infrared sensors below.", None),
    ]),
    ('range', 0.6, 2.6, [
        ("Why go to all this trouble? Because of a quirk in radar physics.", None),
        ("The signal has to travel out and back, so detection range only grows with the fourth root of the radar cross section.", None),
        ("Shrink the echo ten thousand times, and the radar has to be ten times closer to see you.", None),
        ("A wall of overlapping radar coverage turns into scattered islands, with gaps a bomber can slip through.", None),
    ]),
    ('limits', 0.6, 1.4, [
        ("Stealth isn't invisibility.", None),
        ("Shaping works best against short-wavelength radars, like the ones that guide missiles. Long-wavelength early-warning radars can sometimes pick up a faint return.", None),
        ("But on their own, those radars are usually too imprecise to guide a weapon.", None),
        ("So the B-2 also relies on careful route planning, keeping its own radio and radar emissions to a minimum, and flying at night.", None),
    ]),
    ('recap', 0.6, 4.5, [
        ("Shape. Aligned edges. Absorbent skin. Buried engines.", None),
        ("Together, they let a hundred-and-seventy-two-foot bomber slip through air defenses with an echo closer to a bird's than a bomber's.",
         "Together, they let a 172-foot bomber slip through air defenses with an echo closer to a bird's than a bomber's."),
    ]),
]

GAP = 0.45  # silence between lines

def main():
    voice = PiperVoice.load(VOICE)
    cfg = SynthesisConfig(length_scale=1.06, noise_scale=0.6, noise_w_scale=0.8)
    t = 0.0
    tl = {'fps': 30, 'scenes': []}
    n = 0
    for sid, lead, tail, lines in SCENES:
        sc = {'id': sid, 'start': round(t, 3), 'lines': []}
        lt = lead
        for spoken, cap in lines:
            path = os.path.join(OUT, f'{n:02d}_{sid}.wav')
            with wave.open(path, 'wb') as wf:
                voice.synthesize_wav(spoken, wf, syn_config=cfg)
            with wave.open(path, 'rb') as wf:
                dur = wf.getnframes() / wf.getframerate()
            sc['lines'].append({'text': cap or spoken, 'start': round(lt, 3), 'end': round(lt + dur, 3),
                                'wav': os.path.basename(path), 'abs': round(t + lt, 3)})
            lt += dur + GAP
            n += 1
        dur = lt - GAP + tail
        sc['dur'] = round(dur, 3)
        tl['scenes'].append(sc)
        t += dur
        print(f'{sid:8s} start={sc["start"]:7.2f} dur={dur:6.2f}', file=sys.stderr)
    tl['duration'] = round(t, 3)
    with open(os.path.join(HERE, 'timeline.json'), 'w') as f:
        json.dump(tl, f, indent=1)
    print('total', round(t, 2), file=sys.stderr)

if __name__ == '__main__':
    main()
