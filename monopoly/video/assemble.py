"""Join the rendered scenes into the finished video.

    python3 assemble.py            # out/monopoly_decisions.mp4, plus out/chapters.txt
    python3 assemble.py --preview  # also a 720p copy that is easy to send

Video: each scene's render (out/<scene>.mp4), joined with half-second crossfades. Audio, mixed here: the voice-over
files placed at each scene's start, the synthesised sound effects (sfx.py), and audio/music.mp3 underneath if that
file exists, ducked while the narrator speaks. Loudness is normalised for YouTube (-14 LUFS).
"""
import importlib
import os
import subprocess
import sys
import wave

import numpy as np

import sfx
from engine import AUDIO, OUT, SCENE_FILES, Scene
from render import module_for

X = 0.5  # crossfade, seconds
CHAPTERS = {'01_cold_open': 'Five decisions worth $350', '02_the_promise': 'What you control', '03_how_we_tested_it': 'How I tested it', '04_buy_or_pass': 'Decision 1: Buy or pass',
            '05_trading': 'Decision 2: Trading', '06_railroads': 'Decision 3: The railroads', '07a_spend_or_save': 'Decision 4: Spend or save',
            '07b_how_far_to_build': 'How far to build', '07c_which_set_first': 'Two sets: which first?',
            '07d_one_set_or_spread': 'One set, or a house on everything?', '07e_owe_money': 'Owe money? Mortgage first',
            '08_jail': 'Decision 5: Jail', '09_the_computers_player': 'The computer’s own player', '10_the_payoff': 'What it’s worth',
            '11_the_playbook': 'The playbook'}


def probe(path, what='format=duration'):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', what, '-of', 'csv=p=0', path],
                                capture_output=True, text=True, check=True).stdout.strip().split('\n')[0])


def decode(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(sfx.RATE), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def write_wav(path, x):
    st = np.repeat(np.clip(x, -1, 1)[:, None], 2, axis=1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sfx.RATE)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def stamp(t):
    m, s = divmod(int(t), 60)
    return '%d:%02d' % (m, s)


if __name__ == '__main__':
    clips = [os.path.join(OUT, s + '.mp4') for s in SCENE_FILES]
    missing = [c for c in clips if not os.path.exists(c)]
    if missing:
        sys.exit('render these first: ' + ', '.join(os.path.basename(m) for m in missing))
    durs = [probe(c, 'stream=duration') for c in clips]
    starts = [sum(d - X for d in durs[:k]) for k in range(len(durs))]
    total = starts[-1] + durs[-1]
    print('length %s' % stamp(total))

    # audio: narration + effects (+ music)
    n = int((total + 1) * sfx.RATE)
    voice = np.zeros(n, np.float32)
    timed = []
    for k, scene in enumerate(SCENE_FILES):
        v = decode(os.path.join(AUDIO, scene + '.mp3'))
        i = int((starts[k] + Scene.lead_in) * sfx.RATE)
        voice[i:i + len(v)] += v[:n - i]
        sc = importlib.import_module(module_for(scene)).SCENES[scene]()
        timed += [(starts[k] + t, kind) for t, kind in sfx.events(sc)]
        print('  %-26s %s  %d sound cues' % (scene, stamp(starts[k]), len(sfx.events(sc))), flush=True)
    fx = sfx.render_track(timed, total + 1)[:n]
    mix = voice + 0.5 * fx
    music = os.path.join(AUDIO, 'music.mp3')
    if os.path.exists(music):
        m = decode(music)
        m = np.resize(m, n)  # loops if short
        env = np.convolve(np.abs(voice), np.ones(sfx.RATE // 5) / (sfx.RATE // 5), 'same')
        duck = 1 - 0.7 * np.clip(env / 0.05, 0, 1)
        mix += 0.18 * m * duck
    write_wav(os.path.join(OUT, 'mix.wav'), mix * 0.9)

    with open(os.path.join(OUT, 'chapters.txt'), 'w') as f:
        for k, scene in enumerate(SCENE_FILES):
            if scene in CHAPTERS:
                f.write('%s %s\n' % (stamp(starts[k]) if k else '0:00', CHAPTERS[scene]))

    # video: crossfade chain
    cmd = ['ffmpeg', '-y', '-v', 'error', '-stats']
    for c in clips:
        cmd += ['-i', c]
    cmd += ['-i', os.path.join(OUT, 'mix.wav')]
    chain, last = [], '0:v'
    for k in range(1, len(clips)):
        out = 'v%d' % k
        chain.append('[%s][%d:v]xfade=transition=fade:duration=%.3f:offset=%.3f[%s]' % (last, k, X, starts[k], out))
        last = out
    final = os.path.join(OUT, 'monopoly_decisions.mp4')
    cmd += ['-filter_complex', ';'.join(chain), '-map', '[%s]' % last, '-map', '%d:a' % len(clips),
            '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-movflags', '+faststart', '-t', '%.3f' % total, final]
    subprocess.run(cmd, check=True)
    print(final)
    if '--preview' in sys.argv:
        prev = os.path.join(OUT, 'monopoly_decisions_720p.mp4')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', final, '-vf', 'scale=1280:720', '-c:v', 'libx264', '-preset', 'medium',
                        '-crf', '27', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', prev], check=True)
        print(prev)
