#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SysVisual - Panda3D tabanlı "prompt -> video" motoru
=====================================================
Kurulum:   pip install panda3d numpy imageio-ffmpeg
Çalıştır:  python sysvisual.py                      (arayüz)
           python sysvisual.py --prompt "..." --mode fast --out video.mp4

Nasıl çalışır?
  1. Prompt (Türkçe/İngilizce) ayrıştırılır -> Sahne tanımı (mekân, zaman, hava,
     insanlar, hayvanlar, nesneler, aksiyon, kamera, ruh hali, süre)
  2. Panda3D (offscreen) sahneyi prosedürel olarak kurar: arazi, gökyüzü, ışık,
     gölge, sis, yağmur/kar, insanlar (yürür/koşar/dans eder/konuşur), hayvanlar...
  3. Ses motoru: ortam sesleri, ayak sesleri, insan konuşması (formant sentezi),
     köpek/kedi/kuş, araba, ateş, gök gürültüsü, müzik üretilir.
  4. Kareler render edilip ffmpeg ile ses ile birlikte yatay MP4 yapılır.

  SysVisual Lite : 1920x1080 / 30 fps, gölge + antialiasing + yüksek poligon,
                   kareleri tek tek diske akıtır (az RAM, normal süre).
  SysVisual Fast : 1280x720 / 24 fps, tüm CPU çekirdekleri paralel render eder,
                   tüm kareler RAM'de tutulur (çok RAM, çok hızlı).
"""
import os, sys, re, math, time, wave, random, zlib, shutil, subprocess, tempfile
import threading, traceback, argparse
from dataclasses import dataclass, field
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

SR = 44100
QUAL = {
    'lite': dict(W=1920, H=1080, fps=30, seg=20, shadows=True,  aa=True,  part=1.0, preset='slow',     crf='17'),
    'fast': dict(W=1280, H=720,  fps=24, seg=8,  shadows=False, aa=False, part=0.5, preset='veryfast', crf='22'),
}

# ----------------------------------------------------------------------------
# 1) PROMPT ÇÖZÜMLEME
# ----------------------------------------------------------------------------
_T = str.maketrans('ıİşŞğĞüÜöÖçÇ', 'iisSgGuUoOcC')
def fold(s): return s.translate(_T).lower()
def has(t, pat): return re.search(r'\b(?:' + pat + ')', t) is not None

NUMW = {'bir': 1, 'a': 1, 'one': 1, 'iki': 2, 'two': 2, 'uc': 3, 'three': 3, 'dort': 4,
        'four': 4, 'bes': 5, 'five': 5, 'alti': 6, 'six': 6}
COLORS = {'kirmizi': (.8, .1, .1), 'red': (.8, .1, .1), 'mavi': (.15, .3, .85), 'blue': (.15, .3, .85),
          'yesil': (.1, .6, .2), 'green': (.1, .6, .2), 'sari': (.95, .8, .1), 'yellow': (.95, .8, .1),
          'siyah': (.05, .05, .06), 'black': (.05, .05, .06), 'beyaz': (.95, .95, .95), 'white': (.95, .95, .95),
          'turuncu': (.95, .5, .1), 'orange': (.95, .5, .1), 'mor': (.5, .2, .7), 'purple': (.5, .2, .7),
          'pembe': (.95, .5, .7), 'pink': (.95, .5, .7), 'gri': (.5, .5, .5), 'gray': (.5, .5, .5)}
SKIN = [(.96, .8, .69), (.87, .67, .5), (.7, .5, .35), (.45, .3, .2), (.95, .76, .62)]
HAIR = [(.08, .06, .05), (.25, .15, .08), (.5, .35, .15), (.75, .65, .4), (.4, .4, .42)]
SHIRTS = [(.8, .2, .2), (.2, .4, .8), (.2, .6, .3), (.9, .75, .2), (.6, .3, .7), (.9, .5, .2)]

@dataclass
class Scene:
    prompt: str = ''
    seed: int = 1
    setting: str = 'meadow'
    tod: str = 'day'
    weather: str = 'clear'
    mood: str = 'calm'
    camera: str = 'cinematic'
    duration: float = 8.0
    talk: bool = False
    humans: list = field(default_factory=list)
    animals: list = field(default_factory=list)
    props: list = field(default_factory=list)
    action: str = 'walk'
    objects: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    img: dict = field(default_factory=dict)
    cam: dict = field(default_factory=dict)

    def summary(self):
        acts = sorted({h['action'] for h in self.humans})
        return (f"Mekân: {self.setting} | Zaman: {self.tod} | Hava: {self.weather} | Ruh hali: {self.mood}\n"
                f"İnsan: {len(self.humans)} {acts} | Hayvan: {[a['kind'] for a in self.animals]} | "
                f"Nesne: {self.props}\nKamera: {self.camera} | Süre: {self.duration:.0f} sn")

def parse_prompt(prompt):
    t = fold(prompt)
    sc = Scene(prompt=prompt, seed=zlib.crc32(prompt.encode('utf-8')))
    rng = random.Random(sc.seed)
    for name, pat in [('space', 'uzay|space|galaksi|galaxy|gezegen|planet|astronot|astronaut'),
                      ('ocean', 'deniz|okyanus|sahil|plaj|beach|ocean|sea\\b|kumsal'),
                      ('desert', 'col|desert|sahra|kum\\b'),
                      ('snow', 'kar(?:\\b|li|da|in)|snow|kis(?:\\b|in|ta)|winter|buz'),
                      ('city', 'sehir|city|sokak|street|istanbul|new york|london|londra|bina|urban|cadde'),
                      ('forest', 'orman|forest|woods|agac|jungle|tree'),
                      ('meadow', 'cayir|meadow|tarla|field|park|koy|village|bahce|garden')]:
        if has(t, pat):
            sc.setting = name
            break
    if has(t, 'firtina|storm|simsek|thunder'): sc.weather = 'storm'
    elif has(t, 'yagmur|rain|yagis'): sc.weather = 'rain'
    elif has(t, 'sis\\b|sisli|fog|mist'): sc.weather = 'fog'
    elif has(t, 'kar yag|snowfall|snowing') or (sc.setting == 'snow'): sc.weather = 'snow'
    if has(t, 'gece|night|ay isigi|moon|karanlik|dark'): sc.tod = 'night'
    elif has(t, 'gun batimi|aksam|sunset|dusk|evening|golden hour'): sc.tod = 'sunset'
    elif has(t, 'sabah|dawn|sunrise|gun dogumu|morning'): sc.tod = 'dawn'
    if has(t, 'korku|scary|horror|karanlik|dark|gerilim|creepy|urkutucu'): sc.mood = 'scary'
    elif has(t, 'mutlu|happy|neseli|party|parti|fun\\b|eglen|cheer|celebrat'): sc.mood = 'happy'
    elif has(t, 'epik|epic|savas|battle|dramatic|dramatik'): sc.mood = 'epic'
    elif has(t, 'huzun|sad\\b|melankol|melanchol'): sc.mood = 'sad'
    if has(t, 'dron|drone|havadan|aerial|kus bakisi'): sc.camera = 'crane'
    elif has(t, 'yakin|close|portre|portrait'): sc.camera = 'dolly'
    elif has(t, 'takip|follow|tracking'): sc.camera = 'track'
    elif has(t, 'etrafinda|orbit|360|donen'): sc.camera = 'orbit'
    m = re.search(r'(\d+)\s*(?:saniye|sn\b|second|sec\b|s\b)', t)
    if m: sc.duration = float(min(30, max(3, int(m.group(1)))))

    # --- insanlar
    hum = 'insan|adam|kadin|kiz\\b|cocuk|erkek|kisi|people|person|men\\b|man\\b|woman|women|boy|girl|kid|human|astronot|astronaut|karakter|character|lady|anne|baba|mother|father|child'
    n = 0
    mm = re.search(r'\b(\d+|bir|iki|uc|dort|bes|alti|one|two|three|four|five|six|a)\s+(?:\w+\s+)?'
                   r'(?:insan|kisi|adam|kadin|erkek|cocuk|kiz|people|persons?|men|women|kids|children|man|woman|boy|girl|astronauts?|astronot)', t)
    if mm:
        g = mm.group(1); n = int(g) if g.isdigit() else NUMW.get(g, 1)
    elif has(t, hum): n = 1
    n = min(8, n)
    # --- hayvan / nesne
    for kind, pat in [('dog', 'kopek|dog|puppy'), ('cat', 'kedi|cat\\b|kitten'),
                      ('bird', 'kus(?:\\b|lar|u)|bird|karga|crow|eagle|kartal|marti|seagull|pigeon|guvercin')]:
        if has(t, pat): sc.animals.append(dict(kind=kind, R=rng.uniform(4, 7), a0=rng.uniform(0, 6.28),
                                               col=rng.choice([(.55, .35, .15), (.2, .2, .2), (.85, .55, .25), (.8, .8, .78)])))
    for kind, pat in [('car', 'araba|car\\b|cars\\b|taksi|taxi|otomobil|arac|vehicle|truck|kamyon'),
                      ('house', 'ev\\b|evler|house|home\\b|kulube|cabin|cottage'),
                      ('fire', 'ates|fire|kamp|campfire|bonfire|alev')]:
        if has(t, pat): sc.props.append(kind)
    if n == 0 and not sc.animals and 'car' not in sc.props: n = 1
    # --- aksiyon
    act = 'walk'
    if has(t, 'dans|dance|dancing|halay|disko'): act = 'dance'
    elif has(t, 'kos(?:an|uyor|mak|tur|ar)|run|sprint|jog|kacan|escape|chase|kovala'): act = 'run'
    elif has(t, 'el salla|wave|waving|selam|merhaba|hello'): act = 'wave'
    talk = has(t, 'konus|talk|say\\b|says|sohbet|conversation|speak|soyle|diyor|seslen|chat|tartis|argue|sark|sing')
    if talk and act == 'walk' and not has(t, 'yuru|walk|stroll|gez'): act = 'talk'
    sc.talk = bool(talk or act == 'talk')
    cols = [COLORS[m2.group(0)] for m2 in re.finditer('|'.join(r'\b' + c for c in COLORS), t)]
    female = has(t, 'kadin|woman|women|girl|kiz\\b|lady|anne|mother')
    male = has(t, 'adam|erkek|man\\b|men\\b|boy|baba|father')
    child = has(t, 'cocuk|kid|child|boy|girl')
    adult = has(t, 'adam|kadin|erkek|man\\b|men\\b|woman|women|adult|yetiskin')
    for i in range(n):
        g = ('f' if i % 2 == 0 else 'm') if (female and male) else ('f' if female else 'm')
        sc_ = .72 if child and (not adult or i == n - 1) else 1.0
        shirt = cols[i % len(cols)] if cols else SHIRTS[(i + rng.randint(0, 5)) % len(SHIRTS)]
        suit = sc.setting == 'space'
        if suit: shirt = (.9, .9, .92)
        h = dict(gender=g, scale=sc_, shirt=shirt, pants=(.9, .9, .92) if suit else rng.choice([(.15, .17, .3), (.25, .2, .15), (.1, .1, .12)]),
                 skin=rng.choice(SKIN), hair=rng.choice(HAIR), action=act,
                 gait={'walk': .95, 'run': 1.55}.get(act, 1.0), ph0=rng.uniform(0, 6.28), suit=suit, idx=i, n=n)
        if act in ('walk', 'run'):
            h['R'] = max(3.2 if 'fire' in sc.props else 2.5, 3.5 + i * .8); h['a0'] = 2 * math.pi * i / n + rng.uniform(0, .5)
        else:
            h['R'] = 0.0 if n == 1 else (.9 + .12 * n if act == 'talk' else 1.0 + .25 * n)
            h['a0'] = 2 * math.pi * i / max(1, n)
            h['face'] = (math.degrees(h['a0']) + 90) if (act == 'talk' and n > 1) else (180 if n == 1 else rng.uniform(0, 360))
        sc.humans.append(h)
    return sc

try:
    import sysvisual_nlp as _NLP
    def parse_prompt(prompt, image=None, image_mode='both'):
        """Yeni çözümleyici: geniş sözlük, cümle yapısı, koordinatlar, görsel analizi."""
        return _NLP.parse(prompt, Scene, image, image_mode)
except ImportError:
    _NLP = None   # sysvisual_nlp.py yoksa eski basit çözümleyici kullanılır

def speaker_idx(t, n): return int(t / 2.2) % max(1, n)
def car_y(sc, t): return ((np.asarray(t) * 9.0 + (sc.seed % 80)) % 80.0) - 40.0
def flash_times(sc):
    if sc.weather != 'storm': return []
    r = random.Random(sc.seed + 3)
    return sorted(r.uniform(.12, .85) * sc.duration for _ in range(r.randint(2, 3)))

# ----------------------------------------------------------------------------
# 2) SES SENTEZİ
# ----------------------------------------------------------------------------
def bp(x, lo, hi):
    f = np.fft.rfftfreq(len(x), 1.0 / SR); m = np.ones_like(f)
    if hi: m *= 1 / (1 + (f / hi) ** 4)
    if lo: m *= 1 / (1 + (lo / np.maximum(f, 1.0)) ** 4)
    return np.fft.irfft(np.fft.rfft(x) * m, len(x))
def nz(x): return x / (np.std(x) + 1e-9)
VOW = [(730, 1090), (270, 2290), (300, 870), (660, 1720), (520, 1190)]

def synth_audio(sc, path):
    rng = np.random.default_rng(sc.seed); pr = random.Random(sc.seed + 7)
    dur = sc.duration; n = int(dur * SR); t = np.arange(n) / SR
    L = np.zeros(n); R = np.zeros(n)
    def noise(m): return rng.standard_normal(m)
    def put(sig, t0, g=1.0, pan=0.0):
        i = int(t0 * SR)
        if i < 0 or i >= n: return
        sig = sig[:n - i]; a = (pan + 1) * math.pi / 4
        L[i:i + len(sig)] += sig * g * math.cos(a); R[i:i + len(sig)] += sig * g * math.sin(a)
    def bed(fn, g):
        L[:] += g * fn(); R[:] += g * fn()
    st, wt, tod = sc.setting, sc.weather, sc.tod
    acts = {h['action'] for h in sc.humans}
    if st != 'space':
        bed(lambda: nz(bp(noise(n), 50, 450)) * (.55 + .45 * np.sin(2 * np.pi * .07 * t + pr.uniform(0, 6))),
            .10 * (2.2 if wt in ('storm', 'snow') else 1.0))
    else:
        bed(lambda: nz(np.sin(2 * np.pi * 55 * t) + .5 * np.sin(2 * np.pi * 82.4 * t + np.sin(t * .4))), .06)
    if wt in ('rain', 'storm'):
        bed(lambda: nz(bp(noise(n), 1500, 9000)), .10 if wt == 'rain' else .16)
    if st == 'ocean':
        bed(lambda: nz(bp(noise(n), 80, 1200)) * (.5 + .5 * np.sin(2 * np.pi * .11 * t + pr.uniform(0, 6))) ** 2, .30)
    if st == 'city':
        bed(lambda: nz(bp(noise(n), 40, 300)), .12)
    if 'fire' in sc.props:
        bed(lambda: nz(bp(noise(n), 100, 600)), .05)
        for _ in range(int(dur * 18)):
            m = int(pr.uniform(.01, .03) * SR); put(nz(bp(noise(m), 800, 6000)) * np.exp(-np.arange(m) / (m * .3)), pr.uniform(0, dur), pr.uniform(.05, .22), pr.uniform(-.4, .4))
    def chirp(f0, d=.12):
        m = int(d * SR); tt = np.arange(m) / SR
        f = f0 * (1 + .5 * tt / d) + 400 * np.sin(2 * np.pi * 25 * tt)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * tt / d) ** 2
    if tod != 'night' and st in ('meadow', 'forest', 'ocean') and wt in ('clear', 'fog'):
        for _ in range(int(dur * 1.2)):
            t0 = pr.uniform(0, dur); f0 = pr.uniform(2200, 4200); pan = pr.uniform(-.9, .9)
            for k in range(pr.randint(2, 5)): put(chirp(f0 * (1 + .05 * k)), t0 + k * .13, .05, pan)
    if tod == 'night' and st in ('meadow', 'forest', 'desert') and wt in ('clear', 'fog'):
        gate = (np.sin(2 * np.pi * 3 * t) > .3) * (np.sin(2 * np.pi * 33 * t) > 0)
        bed(lambda: np.sin(2 * np.pi * (4300 + pr.uniform(-40, 40)) * t) * gate, .015)
    # ayak sesleri
    ground = {'snow': (1500, 5000, 18), 'desert': (200, 2500, 22), 'city': (300, 3500, 40)}.get(st, (70, 900, 35))
    for i, h in enumerate(sc.humans):
        if h['action'] in ('walk', 'run'):
            step = 1.0 / (2 * h['gait']); tt = pr.uniform(0, step); pan = pr.uniform(-.6, .6)
            while tt < dur:
                m = int(.14 * SR); k = np.arange(m) / SR
                s_ = nz(bp(noise(m), ground[0], ground[1])) * np.exp(-k * ground[2]) + .6 * nz(bp(noise(m), 70, 600)) * np.exp(-k * 40)
                put(s_, tt, .35 if h['action'] == 'walk' else .5, pan); tt += step
    # konuşma
    def voice(f0, length):
        out = []; tc = 0.0; cnt = 0
        while tc < length - .1:
            d = pr.uniform(.09, .22); m = int(d * SR); tt = np.arange(m) / SR
            f = f0 * (1 + .12 * math.sin(tc * 3 + cnt) + pr.uniform(-.05, .05)) * (1 + .08 * np.sin(2 * np.pi * 5 * tt))
            ph = np.cumsum(f) / SR; saw = 2 * (ph % 1) - 1
            F1, F2 = pr.choice(VOW)
            y = bp(saw, F1 * .85, F1 * 1.2) + .7 * bp(saw, F2 * .85, F2 * 1.15) + .1 * nz(bp(noise(m), 2000, 6000))
            out.append(y * np.sin(np.pi * np.arange(m) / m) ** .7)
            gap = pr.uniform(.02, .07) + (.14 if cnt % 4 == 3 else 0); out.append(np.zeros(int(gap * SR)))
            tc += d + gap; cnt += 1
        s_ = np.concatenate(out); return s_ / (np.max(np.abs(s_)) + 1e-9) * .6
    nh = len(sc.humans)
    if sc.talk and nh:
        k = 0
        while k * 2.2 < dur:
            i = k % nh; h = sc.humans[i]
            f0 = 280 if h['scale'] < .8 else (205 if h['gender'] == 'f' else 110) * pr.uniform(.95, 1.08)
            pan = 0.0 if nh == 1 else -.5 + i / max(1, nh - 1)
            put(voice(f0, 1.8), k * 2.2 + .1, .7, pan); k += 1
    # hayvanlar
    for a in sc.animals:
        if a['kind'] == 'dog':
            tt = pr.uniform(.5, 2)
            while tt < dur:
                for b in range(pr.randint(1, 3)):
                    m = int(.2 * SR); k = np.arange(m) / SR
                    f = 480 * np.exp(-k * 3) + 240; ph = np.cumsum(f) / SR; saw = 2 * (ph % 1) - 1
                    y = bp(saw, 500, 900) + .6 * bp(saw, 1400, 2200) + .3 * bp(noise(m), 400, 3000)
                    put(nz(y) * np.exp(-k * 14) * (1 - np.exp(-k * 400)), tt + b * .3, .35, .3)
                tt += pr.uniform(1.5, 4)
        elif a['kind'] == 'cat':
            tt = pr.uniform(1, 3)
            while tt < dur:
                m = int(.9 * SR); k = np.arange(m) / SR; f = 450 + 450 * np.sin(np.pi * k / .9)
                saw = 2 * ((np.cumsum(f) / SR) % 1) - 1
                put(nz(bp(saw, 700, 1100) + .5 * bp(saw, 2200, 2800)) * np.sin(np.pi * k / .9) ** .5, tt, .25, -.3); tt += pr.uniform(3, 6)
        elif a['kind'] == 'bird':
            tt = pr.uniform(0, 1)
            while tt < dur:
                f0 = pr.uniform(2500, 4500)
                for k in range(pr.randint(2, 4)): put(chirp(f0 * (1 + .06 * k), .1), tt + k * .12, .12, pr.uniform(-.7, .7))
                tt += pr.uniform(.8, 2.2)
    # araba
    if 'car' in sc.props:
        grid = np.arange(0, dur + .05, .05); d = np.sqrt(car_y(sc, grid) ** 2 + 12 ** 2)
        amp = np.interp(t, grid, 1 / (1 + d / 5.0))
        saw = 2 * ((np.cumsum(48 + 6 * np.sin(2 * np.pi * 1.3 * t)) / SR) % 1) - 1
        eng = nz(bp(saw, 60, 500) + .3 * nz(bp(noise(n), 100, 900))) * amp * .12
        L[:] += eng * .5; R[:] += eng * .85
        near = np.where(d < 14)[0]
        if len(near):
            m = int(.45 * SR); k = np.arange(m) / SR
            put((np.sin(2 * np.pi * 420 * k) + np.sin(2 * np.pi * 530 * k)) * .5 * np.minimum(1, k * 80) * np.minimum(1, (.45 - k) * 40), grid[near[0]], .15, .6)
    # gök gürültüsü
    for ft in flash_times(sc):
        m = int(3 * SR); k = np.arange(m) / SR
        put(nz(bp(noise(m), 25, 220)) * np.exp(-k / 1.1) * np.minimum(1, k * 30), ft + .4, .9, 0)
    # müzik
    chords = {'calm': [[261.6, 329.6, 392], [220, 261.6, 329.6], [174.6, 220, 261.6], [196, 246.9, 293.7]],
              'happy': [[261.6, 329.6, 392], [293.7, 370, 440], [349.2, 440, 523], [392, 493.9, 587]],
              'scary': [[130.8, 155.6, 185], [123.5, 146.8, 174.6], [130.8, 164.8, 185]],
              'sad': [[220, 261.6, 329.6], [174.6, 207.7, 261.6], [196, 233, 293.7]],
              'epic': [[130.8, 196, 261.6], [155.6, 233, 311], [174.6, 261.6, 349.2], [196, 293.7, 392]]}[sc.mood]
    seg = 3.5
    for k in range(int(dur / seg) + 1):
        m = int(seg * 1.4 * SR); tt = np.arange(m) / SR
        for j, f in enumerate(chords[k % len(chords)]):
            s_ = np.sin(2 * np.pi * f * tt) + .4 * np.sin(4 * np.pi * f * tt) + .2 * np.sin(6 * np.pi * f * tt)
            if sc.mood == 'scary': s_ *= .7 + .3 * np.sin(2 * np.pi * 6 * tt)
            put(s_ * np.sin(np.pi * tt / tt[-1]) ** 2, k * seg, .03 * (1.8 if sc.mood == 'epic' else 1), -.4 + .4 * j)
    if 'dance' in acts or sc.mood == 'happy' and 'dance' in acts:
        tt = 0.0
        while tt < dur:
            m = int(.3 * SR); k = np.arange(m) / SR
            put(np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-k * 30)) / SR) * np.exp(-k * 8), tt, .35, 0)
            m2 = int(.05 * SR); put(nz(bp(noise(m2), 6000, 12000)) * np.exp(-np.arange(m2) / SR * 60), tt + .25, .05, .2)
            tt += .5
    stereo = np.stack([L, R], 1)
    stereo = stereo / (np.max(np.abs(stereo)) + 1e-9) * .89
    fi = int(.25 * SR); fo = int(.5 * SR)
    stereo[:fi] *= np.linspace(0, 1, fi)[:, None]; stereo[-fo:] *= np.linspace(1, 0, fo)[:, None]
    pcm = (stereo * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

# ----------------------------------------------------------------------------
# 3) PANDA3D - PROSEDÜREL GEOMETRİ
# ----------------------------------------------------------------------------
try:
    from panda3d.core import (NodePath, GeomNode, Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
                              GeomTriangles, GeomPoints, GeomLines, AmbientLight, DirectionalLight, PointLight,
                              Fog, Point3, loadPrcFileData, AntialiasAttrib, TransparencyAttrib,
                              OmniBoundingVolume)
except ImportError:
    # Panda3D yüklenemiyorsa (ör. Windows Uygulama Denetimi engeli) web sürümü
    # (sysvisual_web.py) yine de parse_prompt / synth_audio fonksiyonlarını kullanabilir.
    pass

class MB:
    def __init__(s): s.V = []; s.N = []; s.T = []
    def vert(s, p, nrm): s.V.append(p); s.N.append(nrm); return len(s.V) - 1
    def quad(s, a, b, c, d):
        a_, b_, d_ = np.array(a, float), np.array(b, float), np.array(d, float)
        nr = np.cross(b_ - a_, d_ - a_); nr = nr / (np.linalg.norm(nr) + 1e-12)
        i = [s.vert(p, tuple(nr)) for p in (a, b, c, d)]
        s.T += [i[0], i[1], i[2], i[0], i[2], i[3]]
    def node(s, name='m'):
        vd = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic); vd.setNumRows(len(s.V))
        vw = GeomVertexWriter(vd, 'vertex'); nw = GeomVertexWriter(vd, 'normal')
        for p, nr in zip(s.V, s.N): vw.addData3(*p); nw.addData3(*nr)
        pr = GeomTriangles(Geom.UHStatic)
        for i in range(0, len(s.T), 3): pr.addVertices(s.T[i], s.T[i + 1], s.T[i + 2])
        g = Geom(vd); g.addPrimitive(pr); gn = GeomNode(name); gn.addGeom(g); return NodePath(gn)

def box(sx, sy, sz, cz=0.0):
    x, y, z = sx / 2, sy / 2, sz / 2; mb = MB()
    c = lambda i, j, k: (i * x, j * y, k * z + cz)
    for a, b, cc, d in [((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
                        ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)), ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)),
                        ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1))]:
        mb.quad(c(*a), c(*b), c(*cc), c(*d))
    return mb.node('box')

def sphere(r, seg=12):
    mb = MB(); g = []
    for i in range(seg + 1):
        th = math.pi * i / seg; row = []
        for j in range(2 * seg + 1):
            ph = math.pi * j / seg
            nr = (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))
            row.append(mb.vert(tuple(r * v for v in nr), nr))
        g.append(row)
    for i in range(seg):
        for j in range(2 * seg):
            a, b, c, d = g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]
            mb.T += [a, b, c, a, c, d]
    return mb.node('sphere')

def cyl(rt, rb, z0, z1, seg=10, cap=True):
    mb = MB(); lo = []; hi = []
    for j in range(seg + 1):
        a = 2 * math.pi * j / seg; ca, sa = math.cos(a), math.sin(a)
        lo.append(mb.vert((rb * ca, rb * sa, z0), (ca, sa, 0))); hi.append(mb.vert((rt * ca, rt * sa, z1), (ca, sa, 0)))
    for j in range(seg): mb.T += [lo[j], lo[j + 1], hi[j + 1], lo[j], hi[j + 1], hi[j]]
    if cap:
        for z, rad, up in ((z1, rt, 1), (z0, rb, -1)):
            if rad <= 0: continue
            c = mb.vert((0, 0, z), (0, 0, up)); ring = [mb.vert((rad * math.cos(2 * math.pi * j / seg), rad * math.sin(2 * math.pi * j / seg), z), (0, 0, up)) for j in range(seg + 1)]
            for j in range(seg): mb.T += [c, ring[j], ring[j + 1]] if up > 0 else [c, ring[j + 1], ring[j]]
    return mb.node('cyl')

def put(parent, np_, col=None, pos=(0, 0, 0), hpr=(0, 0, 0), scl=1, unlit=False, nofog=False):
    np_.reparentTo(parent); np_.setPos(*pos); np_.setHpr(*hpr)
    np_.setScale(*(scl if isinstance(scl, tuple) else (scl,) * 3))
    if col is not None: np_.setColor(col[0], col[1], col[2], 1)
    if unlit: np_.setLightOff()
    if nofog: np_.setFogOff()
    return np_

def rcol(rng, lo=.3, hi=.9): return tuple(rng.uniform(lo, hi) for _ in range(3))
def mix(a, b, k): return tuple(x * (1 - k) + y * k for x, y in zip(a, b))

def env_for(sc):
    E = {'day': ((.55, .76, .96), (.42, .45, .5), (1, .95, .85), -35, -50), 'sunset': ((.96, .56, .34), (.34, .28, .34), (1, .55, .25), -65, -12),
         'dawn': ((.8, .72, .85), (.38, .36, .44), (1, .82, .7), 60, -14), 'night': ((.02, .03, .08), (.13, .15, .27), (.36, .46, .82), -40, -45)}
    bg, amb, sun, h, p = E[sc.tod]
    fogd = {'clear': .004, 'rain': .012, 'storm': .016, 'fog': .045, 'snow': .012}[sc.weather]
    if sc.weather != 'clear':
        g = sum(bg) / 3; bg = mix(bg, (g * .9,) * 3, .6); sun = tuple(x * .5 for x in sun)
    if sc.mood == 'scary':
        bg = tuple(x * .55 for x in bg); sun = tuple(x * .6 for x in sun); fogd *= 2.2; amb = tuple(x * .8 for x in amb)
    if sc.setting == 'space':
        bg, amb, sun, h, p, fogd = (0, 0, .01), (.1, .1, .12), (1, 1, .96), -50, -25, 0
    return dict(bg=bg, amb=amb, sun=sun, h=h, p=p, fogd=fogd)

# ----------------------------------------------------------------------------
# 4) VARLIKLAR
# ----------------------------------------------------------------------------
class Human:
    def __init__(s, parent, sp, q):
        s.sp = sp; seg = q['seg']; s.root = parent.attachNewNode('human'); s.root.setScale(sp['scale'])
        sk, sh, pa, hr = sp['skin'], sp['shirt'], sp['pants'], sp['hair']
        s.hip = s.root.attachNewNode('hip'); s.hip.setZ(.92)
        put(s.hip, box(.42, .24, .62, .31), sh)
        if sp['gender'] == 'f': put(s.hip, cyl(.2, .31, -.55, .05, 12), sh)
        s.head = s.hip.attachNewNode('head'); s.head.setPos(0, 0, .66)
        put(s.head, sphere(.13, seg), sk, (0, 0, .13))
        put(s.head, sphere(.14, seg), hr, (0, -.015, .17), scl=(1, 1, .8))
        if sp['gender'] == 'f': put(s.head, box(.22, .08, .34, -.1), hr, (0, -.1, .05))
        for sx in (-1, 1): put(s.head, sphere(.018, 6), (.02, .02, .02), (sx * .045, .118, .15))
        if sp['suit']: s.head.attachNewNode('helm'); h_ = put(s.head, sphere(.2, seg), (.6, .8, 1), (0, 0, .13)); h_.setTransparency(TransparencyAttrib.MAlpha); h_.setAlphaScale(.3)
        def limb(x, z, ln, r, top, bot, parent_):
            pv = parent_.attachNewNode('p'); pv.setPos(x, 0, z)
            put(pv, cyl(r, r * .9, -ln, 0, 8), bot); put(pv, cyl(r * 1.05, r, -ln * .5, 0, 8), top); return pv
        s.armL = limb(-.27, .58, .55, .05, sh, sk, s.hip); s.armR = limb(.27, .58, .55, .05, sh, sk, s.hip)
        s.legL = limb(-.1, 0, .9, .075, pa, pa, s.hip); s.legR = limb(.1, 0, .9, .075, pa, pa, s.hip)
        for lg in (s.legL, s.legR): put(lg, box(.11, .22, .07, .03), (.08, .08, .08), (0, .04, -.9))
        a0, R = sp['a0'], sp['R']
        if sp['action'] not in ('walk', 'run'):
            s.root.setPos(R * math.cos(a0), R * math.sin(a0), 0); s.root.setH(sp['face'])
    def update(s, t):
        sp = s.sp; a = sp['action']; ph = 2 * math.pi * sp['gait'] * t + sp['ph0']; sn = math.sin(ph); z = .92
        for nd in (s.armL, s.armR, s.legL, s.legR, s.hip, s.head): nd.setHpr(0, 0, 0)
        if a in ('walk', 'run'):
            spd = 1.4 if a == 'walk' else 3.6; ang = sp['a0'] + spd * t / sp['R']
            s.root.setPos(sp['R'] * math.cos(ang), sp['R'] * math.sin(ang), 0); s.root.setH(math.degrees(ang))
            sw = 32 if a == 'walk' else 58
            s.legL.setP(sn * sw); s.legR.setP(-sn * sw); s.armL.setP(-sn * sw * .8); s.armR.setP(sn * sw * .8)
            z += abs(math.cos(ph)) * (.03 if a == 'walk' else .08); s.hip.setP(-(4 if a == 'walk' else 14))
        elif a == 'dance':
            s.armL.setR(110 + 35 * sn); s.armR.setR(-(110 + 35 * math.cos(ph)))
            s.legL.setP(15 * sn); s.legR.setP(-15 * sn); z += abs(sn) * .06
            s.hip.setR(8 * math.sin(ph * .5)); s.head.setR(10 * math.sin(ph * .5)); s.root.setH(sp['face'] + 15 * math.sin(ph * .5))
        elif a == 'wave':
            s.armR.setR(-(150 + 20 * math.sin(2 * math.pi * 2.5 * t))); s.armL.setP(3 * sn); s.hip.setR(2 * sn)
        else:  # talk
            speaking = speaker_idx(t, sp['n']) == sp['idx']
            s.head.setP((7 if speaking else 2) * math.sin(2 * math.pi * 1.7 * t + sp['ph0']))
            if speaking:
                s.armR.setP(45 + 25 * math.sin(2 * math.pi * .9 * t)); s.armL.setP(20 + 15 * math.sin(2 * math.pi * 1.1 * t))
            else:
                s.armL.setP(3 * sn); s.armR.setP(-3 * sn)
            s.hip.setR(1.5 * sn)
        s.hip.setZ(z)

class Animal:
    def __init__(s, parent, sp, q):
        s.sp = sp; k = sp['kind']; s.k = k; seg = q['seg']; s.root = parent.attachNewNode(k); col = sp['col']; dk = tuple(c * .55 for c in col)
        if k in ('dog', 'cat'):
            s.root.setScale(1.0 if k == 'dog' else .62)
            put(s.root, box(.26, .62, .28, .55), col); put(s.root, sphere(.13, seg), col, (0, .38, .72)); put(s.root, box(.1, .12, .09), dk, (0, .5, .68))
            put(s.root, sphere(.02, 6), (0, 0, 0), (0, .565, .7))
            for sx in (-1, 1): put(s.root, box(.05, .03, .1), dk, (sx * .07, .36, .86))
            s.legs = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pv = s.root.attachNewNode('l'); pv.setPos(sx * .09, sy * .22, .45); put(pv, cyl(.04, .035, -.45, 0, 6), dk); s.legs.append(pv)
            s.tail = s.root.attachNewNode('t'); s.tail.setPos(0, -.32, .65); put(s.tail, cyl(.025, .02, -.3, 0, 6), col)
        else:
            s.root.setScale(1.6)
            put(s.root, sphere(.11, seg), col, scl=(1, 1.8, 1)); put(s.root, sphere(.06, seg), col, (0, .2, .04))
            put(s.root, cyl(0, .025, 0, .08, 6), (.9, .6, .1), (0, .26, .04), (0, -90, 0))
            s.wL = s.root.attachNewNode('wl'); s.wL.setPos(-.07, 0, .03); put(s.wL, box(.5, .14, .015), dk, (-.25, 0, 0))
            s.wR = s.root.attachNewNode('wr'); s.wR.setPos(.07, 0, .03); put(s.wR, box(.5, .14, .015), dk, (.25, 0, 0))
    def update(s, t):
        sp = s.sp
        if s.k == 'bird':
            ang = sp['a0'] + 5.0 * t / (sp['R'] + 3); R = sp['R'] + 3
            s.root.setPos(R * math.cos(ang), R * math.sin(ang), 4 + 1.2 * math.sin(.6 * t + sp['a0'])); s.root.setH(math.degrees(ang)); s.root.setR(-15)
            fl = 40 * math.sin(2 * math.pi * 3 * t); s.wL.setR(fl); s.wR.setR(-fl)
        else:
            spd = 2.4 if s.k == 'dog' else 2.0; ang = sp['a0'] + spd * t / sp['R']; ph = 2 * math.pi * 1.6 * t; sn = math.sin(ph)
            s.root.setPos(sp['R'] * math.cos(ang), sp['R'] * math.sin(ang), abs(sn) * .03); s.root.setH(math.degrees(ang))
            for i, lg in enumerate(s.legs): lg.setP((sn if i in (0, 3) else -sn) * 30)
            s.tail.setHpr(35 * math.sin(ph * 2.2), -120, 0)

class Car:
    def __init__(s, parent, sc, q):
        s.sc = sc; s.root = parent.attachNewNode('car'); rng = random.Random(sc.seed + 11); col = rng.choice([(.75, .1, .1), (.1, .3, .7), (.9, .9, .9), (.9, .7, .1)])
        put(s.root, box(1.8, 4.0, .7, .65), col); put(s.root, box(1.5, 2.0, .6, 1.25), mix(col, (.1, .15, .2), .6), (0, -.2, 0))
        for sx in (-1, 1):
            for sy in (-1.3, 1.3): put(s.root, cyl(.35, .35, -.12, .12, 12), (.05, .05, .05), (sx * .9, sy, .35), (0, 0, 90))
        for sx in (-1, 1): put(s.root, sphere(.13, 8), (1, 1, .8), (sx * .6, 2.0, .7), unlit=True)
        s.root.setX(12)
    def update(s, t): s.root.setY(float(car_y(s.sc, t)))

class Fire:
    def __init__(s, parent, render):
        s.root = parent.attachNewNode('fire')
        for i in range(5): put(s.root, cyl(.08, .08, -.5, .5, 6), (.25, .15, .08), (0, 0, .1), (i * 72, 80, 0))
        s.fl = [put(s.root, cyl(0, .22 - i * .05, .1, .9 + i * .3, 8), c, unlit=True) for i, c in enumerate([(1, .45, .05), (1, .75, .15), (1, .95, .5)])]
        pl = PointLight('fire'); pl.setColor((1.4, .65, .25, 1)); pl.setAttenuation((1, 0, .06)); s.pl = pl
        s.plnp = s.root.attachNewNode(pl); s.plnp.setPos(0, 0, 1.0); render.setLight(s.plnp)
    def update(s, t):
        for i, f in enumerate(s.fl): f.setScale(1 + .15 * math.sin(t * (11 + i * 4) + i), 1 + .15 * math.cos(t * (9 + i * 3)), 1 + .25 * math.sin(t * (13 + i * 5) + 2))
        k = 1 + .25 * math.sin(t * 17) * math.sin(t * 5.3); s.pl.setColor((1.4 * k, .65 * k, .25 * k, 1))

class Particles:
    def __init__(s, render, kind, q, seed):
        rng = np.random.default_rng(seed); s.kind = kind
        n = int({'rain': 2500, 'snow': 1800}[kind] * q['part']) * (1.5 if kind == 'rain' and False else 1)
        s.n = n; s.box = np.array([36.0, 36.0, 16.0]); s.base = rng.random((n, 3)) * s.box
        s.vel = np.zeros((n, 3)); s.vel[:, 2] = -(18.0 if kind == 'rain' else 1.6) * (0.85 + .3 * rng.random(n))
        s.vel[:, 0] = .15 * (-s.vel[:, 2]) * .15 if kind == 'rain' else 0
        s.ph = rng.random(n) * 6.28; npts = n * (2 if kind == 'rain' else 1)
        s.vd = GeomVertexData('p', GeomVertexFormat.getV3(), Geom.UHDynamic); s.vd.setNumRows(npts)
        pr = GeomLines(Geom.UHStatic) if kind == 'rain' else GeomPoints(Geom.UHStatic); pr.addNextVertices(npts)
        g = Geom(s.vd); g.addPrimitive(pr); gn = GeomNode('part'); gn.addGeom(g); s.np = render.attachNewNode(gn)
        s.np.node().setBounds(OmniBoundingVolume()); s.np.node().setFinal(True); s.np.setLightOff(); s.np.setShaderOff()
        if kind == 'rain': s.np.setTransparency(TransparencyAttrib.MAlpha); s.np.setColor(.8, .85, .95, .45)
        else: s.np.setRenderModeThickness(4); s.np.setColor(1, 1, 1, 1)
    def update(s, t, cam):
        p = (s.base + s.vel * t) % s.box; p[:, 0] -= s.box[0] / 2; p[:, 1] -= s.box[1] / 2; p[:, 0] += cam.x; p[:, 1] += cam.y
        if s.kind == 'snow': p[:, 0] += .4 * np.sin(t * .8 + s.ph); arr = p
        else:
            p2 = p - np.array([.1, 0, -.55]) * 1.0; arr = np.empty((s.n * 2, 3)); arr[0::2] = p; arr[1::2] = p2
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        try:
            mv = memoryview(s.vd.modifyArray(0)).cast('B').cast('f'); mv[:] = memoryview(arr.reshape(-1))
        except Exception:
            w = GeomVertexWriter(s.vd, 'vertex')
            for v in arr: w.setData3(float(v[0]), float(v[1]), float(v[2]))

# ----------------------------------------------------------------------------
# 5) DÜNYA + KAMERA
# ----------------------------------------------------------------------------
def make_tree(parent, kind, h, leaf, seg):
    n = parent.attachNewNode('tree'); put(n, cyl(.18 * h / 5, .25 * h / 5, 0, h * .35, 6), (.35, .22, .1))
    if kind == 'pine':
        for i in range(3): put(n, cyl(0, (1.6 - i * .4) * h / 5, h * (.25 + i * .22), h * (.25 + i * .22) + h * .45, seg), leaf)
    else: put(n, sphere(h * .28, seg), leaf, (0, 0, h * .62), scl=(1, 1, .9))
    return n

class World:
    def __init__(s, base, sc, q):
        s.sc = sc; s.q = q; s.render = base.render; rng = random.Random(sc.seed + 1); s.rng = rng
        s.env = env_for(sc); e = s.env; seg = q['seg']; r = s.render.attachNewNode('world'); s.root = r
        base.setBackgroundColor(*e['bg'], 1)
        amb = AmbientLight('a'); amb.setColor((*e['amb'], 1)); s.ambnp = s.render.attachNewNode(amb); s.render.setLight(s.ambnp); s.amb_c = e['amb']
        dl = DirectionalLight('d'); dl.setColor((*e['sun'], 1)); s.dlnp = s.render.attachNewNode(dl); s.dlnp.setHpr(e['h'], e['p'], 0); s.render.setLight(s.dlnp)
        h_, p_ = math.radians(e['h']), math.radians(e['p']); fwd = (-math.sin(h_) * math.cos(p_), math.cos(h_) * math.cos(p_), math.sin(p_)); s.sundir = tuple(-v for v in fwd)
        if q['shadows']:
            s.dlnp.setPos(*(v * 60 for v in s.sundir)); dl.setShadowCaster(True, 2048, 2048); ln = dl.getLens(); ln.setFilmSize(45, 45); ln.setNearFar(5, 160); s.render.setShaderAuto()
        if q['aa']: s.render.setAntialias(AntialiasAttrib.MAuto)
        if e['fogd'] > 0:
            fg = Fog('f'); fg.setColor(*e['bg']); fg.setExpDensity(e['fogd']); s.render.setFog(fg)
        s.build_sky(sc, rng, seg); s.build_setting(sc, rng, seg)
        s.ents = []; s.subjects = []; s.fire = None; s.car = None
        for hsp in sc.humans: h = Human(r, hsp, q); s.ents.append(h); s.subjects.append(h)
        for a in sc.animals:
            an = Animal(r, a, q); s.ents.append(an)
            if a['kind'] != 'bird': s.subjects.append(an)
        if 'car' in sc.props: s.car = Car(r, sc, q); s.ents.append(s.car)
        if 'fire' in sc.props: s.fire = Fire(r, s.render); s.ents.append(s.fire)
        if 'house' in sc.props:
            hs = r.attachNewNode('house'); hs.setPos(-14, 16, 0); hs.setH(150)
            put(hs, box(5, 6, 2.8, 1.4), (.75, .65, .5)); put(hs, cyl(0, 4.8, 2.8, 4.6, 4), (.55, .2, .15), hpr=(45, 0, 0)); put(hs, box(.9, .1, 2, 1), (.3, .18, .1), (0, -3.02, 0))
            wc = (1, .85, .45) if sc.tod == 'night' else (.4, .55, .7)
            for sx in (-1.6, 1.6): put(hs, box(.9, .1, .9, 0), wc, (sx, -3.03, 1.5), unlit=(sc.tod == 'night'))
        s.parts = None
        if sc.weather in ('rain', 'storm'): s.parts = Particles(s.render, 'rain', q, sc.seed)
        elif sc.weather == 'snow': s.parts = Particles(s.render, 'snow', q, sc.seed)
        s.flashes = flash_times(sc)
    def build_sky(s, sc, rng, seg):
        r = s.render.attachNewNode('sky'); e = s.env
        if sc.tod == 'night' or sc.setting == 'space':
            vd = GeomVertexData('st', GeomVertexFormat.getV3(), Geom.UHStatic); N = 900; vd.setNumRows(N); w = GeomVertexWriter(vd, 'vertex')
            for _ in range(N):
                a, b = rng.uniform(0, 6.283), rng.uniform(.05, 1.4); w.addData3(900 * math.cos(a) * math.cos(b), 900 * math.sin(a) * math.cos(b), 900 * math.sin(b))
            pr = GeomPoints(Geom.UHStatic); pr.addNextVertices(N); g = Geom(vd); g.addPrimitive(pr); gn = GeomNode('stars'); gn.addGeom(g)
            sn = r.attachNewNode(gn); sn.setRenderModeThickness(2); sn.setLightOff(); sn.setFogOff(); sn.setColor(1, 1, 1, 1); sn.node().setBounds(OmniBoundingVolume()); sn.node().setFinal(True)
        if sc.setting == 'space':
            put(r, sphere(60, 24), (.2, .45, .75), (-120, 250, 90), unlit=True, nofog=True)
            put(r, sphere(14, 16), (.85, .85, .85), (200, 300, 120), unlit=True, nofog=True)
        d = tuple(v * 400 for v in s.sundir)
        if sc.setting != 'space':
            if sc.tod == 'night': put(r, sphere(14, 16), (.9, .92, 1), d, unlit=True, nofog=True)
            elif sc.weather == 'clear': put(r, sphere(18, 16), (1, .97, .8), d, unlit=True, nofog=True)
            if sc.tod != 'night':
                for _ in range(7):
                    a = rng.uniform(0, 6.28); dist = rng.uniform(150, 280); z = rng.uniform(55, 90); gc = .96 if sc.weather == 'clear' else .55
                    for k in range(3): put(r, sphere(rng.uniform(12, 20), 8), (gc, gc, gc), (dist * math.cos(a) + k * 14, dist * math.sin(a), z), scl=(1.6, 1, .5), unlit=True, nofog=True)
    def ring(s, n, rmin, rmax):
        return [(math.cos(a) * d, math.sin(a) * d) for a, d in ((s.rng.uniform(0, 6.28), s.rng.uniform(rmin, rmax)) for _ in range(n))]
    def build_setting(s, sc, rng, seg):
        r = s.root; st = sc.setting
        gcol = {'meadow': (.28, .5, .2), 'forest': (.18, .34, .14), 'desert': (.82, .68, .42), 'snow': (.92, .94, .98), 'city': (.3, .3, .32), 'ocean': (.1, .35, .55), 'space': (.45, .44, .43)}[st]
        put(r, cyl(300, 300, -.5, 0, 48), gcol)
        if st == 'meadow':
            for _ in range(5): put(r, sphere(1, 12), (.22, .45, .18), (rng.uniform(-150, 150), rng.uniform(80, 200), 0), scl=(rng.uniform(30, 60), rng.uniform(30, 50), rng.uniform(8, 16)))
            for x, y in s.ring(70, 3, 30): put(r, sphere(.07, 6), rng.choice([(1, .9, .2), (.95, .3, .4), (1, 1, 1), (.6, .4, .9)]), (x, y, .12))
            for x, y in s.ring(14, 12, 60): make_tree(r, 'round', rng.uniform(5, 8), (.15, .42 + rng.uniform(0, .15), .15), seg).setPos(x, y, 0)
        elif st == 'forest':
            for x, y in s.ring(55, 9, 70): make_tree(r, rng.choice(['pine', 'round']), rng.uniform(6, 11), (.1, .3 + rng.uniform(0, .15), .12), seg).setPos(x, y, 0)
            for x, y in s.ring(8, 4, 20): put(r, cyl(.2, .2, 0, 3, 8), (.3, .2, .1), (x, y, .2), (rng.uniform(0, 360), 90, 0))
        elif st == 'desert':
            for _ in range(10): a = rng.uniform(0, 6.28); d = rng.uniform(40, 120); put(r, sphere(1, 12), (.8, .65, .4), (d * math.cos(a), d * math.sin(a), 0), scl=(rng.uniform(20, 40), rng.uniform(20, 35), rng.uniform(4, 8)))
            for x, y in s.ring(9, 8, 40):
                c = r.attachNewNode('cactus'); c.setPos(x, y, 0); put(c, cyl(.25, .25, 0, 2.6, 8), (.2, .5, .2)); put(c, cyl(.18, .18, 0, 1, 8), (.2, .5, .2), (.5, 0, 1.2), (0, 0, 0))
            for x, y in s.ring(10, 6, 40): put(r, sphere(rng.uniform(.3, .8), 6), (.5, .42, .35), (x, y, .1))
        elif st == 'snow':
            for x, y in s.ring(28, 9, 60):
                tr = make_tree(r, 'pine', rng.uniform(6, 10), (.12, .35, .2), seg); tr.setPos(x, y, 0)
            for x, y in s.ring(10, 3, 25): put(r, sphere(rng.uniform(.4, 1), 6), (.95, .97, 1), (x, y, .05), scl=(1, 1, .6))
            sm = r.attachNewNode('snowman'); sm.setPos(7, -7, 0)
            for z, rad in ((.5, .5), (1.3, .38), (1.9, .27)): put(sm, sphere(rad, seg), (.97, .97, 1), (0, 0, z))
            put(sm, cyl(0, .05, 0, .3, 6), (.95, .45, .1), (0, .27, 1.9), (0, -90, 0))
        elif st == 'ocean':
            put(r, cyl(13, 13, -.4, .02, 40), (.85, .77, .55)); s.waves = []
            for i in range(24):
                a = i / 24 * 6.2832 * 3 % 6.2832; w_ = put(r, box(8, .8, .25), (.9, .95, 1), hpr=(math.degrees(a) + 90, 0, 0)); s.waves.append((w_, a, i / 24))
            for x, y in s.ring(5, 5, 11):
                p = r.attachNewNode('palm'); p.setPos(x, y, 0); put(p, cyl(.12, .2, 0, 5, 6), (.4, .28, .15), hpr=(0, 0, 8))
                for k in range(6): put(p, cyl(0, .5, 0, 3, 4), (.15, .5, .2), (0, 0, 5), (k * 60, 70, 0), scl=(1, 1, 1))
        elif st == 'city':
            put(r, box(9, 400, .05, 0), (.12, .12, .13), (0, 0, .03))
            for k in range(-20, 20): put(r, box(.2, 2, .02, 0), (.9, .9, .85), (0, k * 5, .07))
            for side in (-1, 1):
                for k in range(-7, 8):
                    w = rng.uniform(8, 13); d = rng.uniform(9, 14); hg = rng.uniform(8, 32); x = side * (9 + w / 2 + rng.uniform(0, 2)); y = k * 15 + rng.uniform(-2, 2)
                    put(r, box(w, d, hg, hg / 2), rcol(rng, .35, .6), (x, y, 0))
                    mb = MB(); xf = x - side * (w / 2 + .03)
                    for ci in range(int(d / 2.6)):
                        for ri in range(int(hg / 3.2)):
                            if sc.tod == 'night' and rng.random() > .55: continue
                            y0 = y - d / 2 + ci * 2.6 + .5; z0 = 1.5 + ri * 3.2
                            mb.quad((xf, y0, z0), (xf, y0 + 1.5, z0), (xf, y0 + 1.5, z0 + 1.6), (xf, y0, z0 + 1.6))
                    if mb.V: put(r, mb.node('win'), (1, .85, .5) if sc.tod == 'night' else (.25, .35, .5), unlit=True)
            for side in (-1, 1):
                for k in range(-8, 9, 2):
                    put(r, cyl(.08, .08, 0, 5, 6), (.2, .2, .2), (side * 5.2, k * 7, 0)); put(r, sphere(.25, 8), (1, .9, .6) if sc.tod == 'night' else (.8, .8, .8), (side * 5.2, k * 7, 5.1), unlit=True)
        elif st == 'space':
            for _ in range(24): a = rng.uniform(0, 6.28); d = rng.uniform(8, 100); put(r, sphere(1, 8), (.3, .3, .3), (d * math.cos(a), d * math.sin(a), 0), scl=(rng.uniform(1, 6), rng.uniform(1, 6), .08))
            for x, y in s.ring(15, 6, 60): put(r, sphere(rng.uniform(.4, 1.6), 6), (.4, .38, .36), (x, y, .1))
    def update_ents(s, t):
        for e in s.ents: e.update(t)
        if hasattr(s, 'waves'):
            for w_, a, ph in s.waves:
                rr = 40 - ((t * 1.2 + ph * 26) % 26); w_.setPos(rr * math.cos(a), rr * math.sin(a), -.05 + .12 * math.sin(t * 1.3 + ph * 9))
        c = s.amb_c
        if any(f < t < f + .12 for f in s.flashes): c = (.9, .9, 1)
        s.ambnp.node().setColor((*c, 1))
    def focus(s):
        if s.subjects:
            ps = [e.root.getPos(s.render) for e in s.subjects]; x = sum(p.x for p in ps) / len(ps); y = sum(p.y for p in ps) / len(ps)
            return Point3(x, y, 1.1 if s.sc.humans else .5)
        return Point3(0, 0, 1.2)

class Director:
    def __init__(s, world, sc):
        s.w = world; rng = random.Random(sc.seed + 5); dur = sc.duration
        if sc.camera == 'cinematic':
            n = max(1, min(5, round(dur / 3.5))); modes = ['orbit', 'dolly', 'track', 'low', 'crane', 'orbit']; rng.shuffle(modes)
        else: n = 1; modes = [sc.camera]
        b = [dur * i / n for i in range(n + 1)]; s.shots = []
        for i in range(n):
            s.shots.append(dict(t0=b[i], t1=b[i + 1], mode=modes[i % len(modes)], a0=rng.uniform(0, 6.28), w=rng.choice([-1, 1]) * rng.uniform(.12, .3), r=rng.uniform(5, 8),
                                h=rng.uniform(1.2, 2.6), r0=rng.uniform(9, 12), r1=rng.uniform(2.8, 4), h1=rng.uniform(8, 14), off=rng.uniform(-.7, .7), d=rng.uniform(3, 4.5)))
    def apply(s, cam, t):
        sh = next((x for x in s.shots if t < x['t1']), s.shots[-1]); u = min(1, max(0, (t - sh['t0']) / (sh['t1'] - sh['t0'] + 1e-9))); su = u * u * (3 - 2 * u)
        f = s.w.focus(); m = sh['mode']; a0 = sh['a0']; cs, sn = math.cos, math.sin
        if m == 'dolly': r = sh['r0'] + (sh['r1'] - sh['r0']) * su; p = (f.x + r * cs(a0), f.y + r * sn(a0), f.z + .4 + (1 - su) * 1.5)
        elif m == 'crane': a = a0 + .25 * u; p = (f.x + 10 * cs(a), f.y + 10 * sn(a), 1.0 + (sh['h1'] - 1.0) * su)
        elif m == 'track' and s.w.subjects:
            hd = math.radians(s.w.subjects[0].root.getH()); bx, by = sn(hd), -cs(hd); ca, sa = cs(sh['off']), sn(sh['off'])
            p = (f.x + (bx * ca - by * sa) * sh['d'], f.y + (bx * sa + by * ca) * sh['d'], f.z + .5)
        elif m == 'low': a = a0 + .06 * (t - sh['t0']); p = (f.x + (sh['r'] + 2) * cs(a), f.y + (sh['r'] + 2) * sn(a), .45 + .3 * su)
        else: a = a0 + sh['w'] * (t - sh['t0']) * 2; p = (f.x + sh['r'] * cs(a), f.y + sh['r'] * sn(a), f.z + sh['h'] - .2)
        p = (p[0] + .015 * sn(t * 7.3), p[1] + .015 * sn(t * 5.1), max(.3, p[2] + .015 * sn(t * 6.2)))
        cam.setPos(*p); cam.lookAt(f)

# ----------------------------------------------------------------------------
# 6) RENDER HATTI
# ----------------------------------------------------------------------------
def render_frames(sc, mode, ids, sink, tick):
    q = QUAL[mode]; W, H = q['W'], q['H']; soft = os.environ.get('SYSVISUAL_SOFT') == '1'
    loadPrcFileData('', f"window-type offscreen\nwin-size {W} {H}\naudio-library-name null\nsync-video 0\nshow-frame-rate-meter 0\n"
                        f"framebuffer-multisample {1 if q['aa'] else 0}\nmultisamples 4\nload-display {'p3tinydisplay' if soft else 'pandagl'}\n")
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase(); base.disableMouse(); base.camLens.setFov(70); base.camLens.setNearFar(.1, 2500)
    world = World(base, sc, q); director = Director(world, sc); cam = base.camera
    for _ in range(2): base.graphicsEngine.renderFrame()
    for c, i in enumerate(ids, 1):
        t = i / q['fps']; world.update_ents(t); director.apply(cam, t)
        if world.parts: world.parts.update(t, cam.getPos())
        base.graphicsEngine.renderFrame()
        tex = base.win.getScreenshot(); data = np.frombuffer(bytes(tex.getRamImageAs('RGB')), np.uint8)
        arr = np.ascontiguousarray(data.reshape(tex.getYSize(), tex.getXSize(), 3)[::-1])
        if arr.shape != (H, W, 3): raise RuntimeError(f'Beklenmeyen kare boyutu {arr.shape}')
        sink(i, arr); tick(c)
    base.destroy()

def find_ffmpeg():
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        p = shutil.which('ffmpeg')
        if p: return p
    raise RuntimeError("ffmpeg bulunamadı. Kur:  pip install imageio-ffmpeg")

def ff_cmd(ff, q, wav, out):
    return [ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{q['W']}x{q['H']}", '-r', str(q['fps']), '-i', '-', '-i', wav,
            '-c:v', 'libx264', '-preset', q['preset'], '-crf', q['crf'], '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out]

def lite_worker(sc, out, wav, ff, pq):
    try:
        q = QUAL['lite']; N = int(sc.duration * q['fps']); p = subprocess.Popen(ff_cmd(ff, q, wav, out), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        render_frames(sc, 'lite', range(N), lambda i, a: p.stdin.write(a.tobytes()), lambda c: pq.put(('p', 1)))
        p.stdin.close(); p.wait(); pq.put(('wdone', None))
    except Exception: pq.put(('err', traceback.format_exc()))

def fast_worker(sc, ids, shm_name, shape, pq):
    try:
        shm = shared_memory.SharedMemory(name=shm_name); arr = np.ndarray(shape, np.uint8, buffer=shm.buf)
        def sink(i, a): arr[i] = a
        render_frames(sc, 'fast', ids, sink, lambda c: pq.put(('p', 1)))
        del arr; shm.close(); pq.put(('wdone', None))
    except Exception: pq.put(('err', traceback.format_exc()))

def generate(prompt, mode, out_path, progress=lambda f, m: None, log=lambda m: None):
    sc = parse_prompt(prompt); q = QUAL[mode]; N = int(sc.duration * q['fps']); log(sc.summary())
    tmp = tempfile.mkdtemp(prefix='sysvisual_'); wav = os.path.join(tmp, 'a.wav'); ff = find_ffmpeg()
    progress(.02, 'Ses sentezleniyor...'); synth_audio(sc, wav); progress(.05, 'Sahne kuruluyor...')
    ctx = mp.get_context('spawn'); pq = ctx.Queue(); procs = []; shm = None
    try:
        if mode == 'lite':
            procs = [ctx.Process(target=lite_worker, args=(sc, out_path, wav, ff, pq))]
        else:
            nw = max(1, min(os.cpu_count() or 2, 8, N)); shape = (N, q['H'], q['W'], 3); size = int(np.prod(shape))
            log(f'Fast modu: {nw} işçi, ~{size / 1e9:.2f} GB RAM ayrılıyor')
            shm = shared_memory.SharedMemory(create=True, size=size); chunks = np.array_split(np.arange(N), nw)
            procs = [ctx.Process(target=fast_worker, args=(sc, [int(x) for x in c], shm.name, shape, pq)) for c in chunks if len(c)]
        for p in procs: p.start()
        done = 0; fr = 0; span = .95 if mode == 'lite' else .8
        while done < len(procs):
            try: k, v = pq.get(timeout=1)
            except Exception:
                if not any(p.is_alive() for p in procs) and pq.empty(): raise RuntimeError('Render işçileri beklenmedik şekilde kapandı (GPU/ekran sürücüsünü kontrol et; SYSVISUAL_SOFT=1 dene).')
                continue
            if k == 'err': raise RuntimeError(v)
            if k == 'p': fr += 1; progress(.05 + span * fr / N, f'Render: kare {fr}/{N}')
            elif k == 'wdone': done += 1
        for p in procs: p.join()
        if mode == 'fast':
            progress(.86, 'MP4 kodlanıyor...'); arr = np.ndarray(shape, np.uint8, buffer=shm.buf)
            pr = subprocess.Popen(ff_cmd(ff, q, wav, out_path), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for i in range(N): pr.stdin.write(arr[i].tobytes()); progress(.86 + .14 * i / N, 'MP4 kodlanıyor...')
            pr.stdin.close(); pr.wait(); del arr
        progress(1.0, 'Tamamlandı'); log(f'Video hazır: {out_path}'); return out_path
    finally:
        for p in procs:
            if p.is_alive(): p.terminate()
        if shm is not None:
            try: shm.close(); shm.unlink()
            except Exception: pass
        shutil.rmtree(tmp, ignore_errors=True)

# ----------------------------------------------------------------------------
# 7) ARAYÜZ
# ----------------------------------------------------------------------------
def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import queue
    root = tk.Tk(); root.title('SysVisual - Panda3D Video Yapay Zekası'); root.geometry('860x640'); root.configure(bg='#14161c')
    uiq = queue.Queue(); state = {'out': None, 'busy': False}
    tk.Label(root, text='SysVisual', font=('Segoe UI', 22, 'bold'), fg='#7cc4ff', bg='#14161c').pack(anchor='w', padx=18, pady=(14, 0))
    tk.Label(root, text='Ne görmek istediğini yaz; Panda3D sahneyi kurup videoya çevirsin.', fg='#9aa4b2', bg='#14161c').pack(anchor='w', padx=18)
    txt = tk.Text(root, height=5, wrap='word', font=('Segoe UI', 12), bg='#1d212b', fg='#e8edf5', insertbackground='white', relief='flat', padx=10, pady=8)
    txt.pack(fill='x', padx=18, pady=10)
    txt.insert('1.0', 'Gün batımında karlı bir ormanda 3 kişi sohbet ediyor, yanlarında köpek var, ateş yanıyor, 12 saniye')
    row = tk.Frame(root, bg='#14161c'); row.pack(fill='x', padx=18)
    mode = tk.StringVar(value='lite')
    for val, label in (('lite', 'SysVisual Lite  (1080p/30, ayrıntılı)'), ('fast', 'SysVisual Fast  (720p/24, çok hızlı, yüksek RAM)')):
        tk.Radiobutton(row, text=label, variable=mode, value=val, fg='#e8edf5', bg='#14161c', selectcolor='#1d212b', activebackground='#14161c', activeforeground='white').pack(side='left', padx=(0, 18))
    send = tk.Button(row, text='Gönder ▶', bg='#2f81f7', fg='white', relief='flat', padx=18, pady=6, font=('Segoe UI', 11, 'bold')); send.pack(side='right')
    bar = ttk.Progressbar(root, maximum=1.0); bar.pack(fill='x', padx=18, pady=(14, 2))
    status = tk.Label(root, text='Hazır', fg='#9aa4b2', bg='#14161c', anchor='w'); status.pack(fill='x', padx=18)
    logb = tk.Text(root, height=12, bg='#0e1015', fg='#b8c2d0', relief='flat', state='disabled', font=('Consolas', 10)); logb.pack(fill='both', expand=True, padx=18, pady=10)
    row2 = tk.Frame(root, bg='#14161c'); row2.pack(fill='x', padx=18, pady=(0, 14))
    play = tk.Button(row2, text='Oynat', state='disabled', relief='flat', padx=14, pady=5); save = tk.Button(row2, text='MP4 İndir', state='disabled', relief='flat', padx=14, pady=5)
    play.pack(side='left'); save.pack(side='left', padx=8)
    def addlog(m): logb.configure(state='normal'); logb.insert('end', m + '\n'); logb.see('end'); logb.configure(state='disabled')
    def job(prompt, md):
        out = os.path.join(tempfile.gettempdir(), f'sysvisual_{int(time.time())}.mp4')
        try:
            generate(prompt, md, out, lambda f, m: uiq.put(('p', f, m)), lambda m: uiq.put(('l', m))); uiq.put(('ok', out))
        except Exception as e: uiq.put(('e', str(e)))
    def on_send():
        p = txt.get('1.0', 'end').strip()
        if not p or state['busy']: return
        state['busy'] = True; send.configure(state='disabled'); play.configure(state='disabled'); save.configure(state='disabled'); bar['value'] = 0
        threading.Thread(target=job, args=(p, mode.get()), daemon=True).start()
    def poll():
        try:
            while True:
                m = uiq.get_nowait()
                if m[0] == 'p': bar['value'] = m[1]; status.configure(text=m[2])
                elif m[0] == 'l': addlog(m[1])
                elif m[0] == 'ok': state.update(out=m[1], busy=False); send.configure(state='normal'); play.configure(state='normal'); save.configure(state='normal'); status.configure(text='Video hazır ✔')
                elif m[0] == 'e': state['busy'] = False; send.configure(state='normal'); status.configure(text='Hata'); addlog(m[1]); messagebox.showerror('SysVisual', m[1][:1500])
        except queue.Empty: pass
        root.after(100, poll)
    def on_play():
        if not state['out']: return
        if sys.platform.startswith('win'): os.startfile(state['out'])
        elif sys.platform == 'darwin': subprocess.Popen(['open', state['out']])
        else: subprocess.Popen(['xdg-open', state['out']])
    def on_save():
        if not state['out']: return
        d = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[('MP4 video', '*.mp4')], initialfile='sysvisual.mp4')
        if d: shutil.copy(state['out'], d); addlog(f'Kaydedildi: {d}')
    send.configure(command=on_send); play.configure(command=on_play); save.configure(command=on_save)
    poll(); root.mainloop()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prompt'); ap.add_argument('--mode', choices=['lite', 'fast'], default='lite'); ap.add_argument('--out', default='sysvisual.mp4')
    a = ap.parse_args()
    if a.prompt: generate(a.prompt, a.mode, a.out, lambda f, m: print(f'\r{int(f * 100):3d}% {m}   ', end='', flush=True), print); print()
    else: gui()

if __name__ == '__main__':
    mp.freeze_support(); main()
