import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Mic, MicOff, Play, Search, Square, X } from "lucide-react";
import { PitchDetector } from "pitchy";
import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

type Props = {
  onBack: () => void;
  isDarkMode: boolean;
  user: any;
  initialSongId?: number | string;
  initialSongTitle?: string;
  initialSongArtist?: string;
};

type Song = {
  song_id: number;
  title: string;
  artist: string;
  midi_min: number;
  midi_median: number;
  midi_max: number;
};

type PitchFrames = {
  hop_ms: number;
  times: number[];
  hz: (number | null)[];
};

function hzToMidi(hz: number): number {
  return 69 + 12 * Math.log2(hz / 440);
}

function midiToNoteName(midi: number): string {
  const names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
  const oct = Math.floor(midi / 12) - 1;
  return names[midi % 12] + oct;
}

export default function AccompanimentPage({ onBack, isDarkMode, user, initialSongId, initialSongTitle, initialSongArtist }: Props) {
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSong, setSelectedSong] = useState<Song | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [showList, setShowList] = useState(false);
  const [semitones, setSemitones] = useState(0);
  const [pitchFrames, setPitchFrames] = useState<PitchFrames | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [isEchoOn, setIsEchoOn] = useState(false);
  const [userPitch, setUserPitch] = useState<number | null>(null);
  const echoGainRef = useRef<GainNode | null>(null);
  const semitonesRef = useRef(semitones);
  const [originalPitch, setOriginalPitch] = useState<number | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const accompCtxRef = useRef<AudioContext | null>(null);
  const accompSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const canvasRafRef = useRef<number | null>(null);
  const sampleIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<ReturnType<typeof PitchDetector.forFloat32Array> | null>(null);
  const micBufRef = useRef<Float32Array | null>(null);

  const [songEnded, setSongEnded] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [accompanimentModel, setAccompanimentModel] = useState<"kim" | "mdx">("kim");

  type LogEntry = { time: number; userMidi: number | null; origMidi: number | null };
  const fullSongLog = useRef<LogEntry[]>([]);

  // 피치 히스토리 (Canvas용)
  const userPitchHistory = useRef<{ time: number; midi: number | null }[]>([]);
  const origPitchHistory = useRef<(number | null)[]>([]);
  const MAX_HISTORY = 60;
  const pitchFramesRef = useRef<PitchFrames | null>(null);
  const selectedSongRef = useRef<Song | null>(null);

  const dark = isDarkMode;
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";
  const textColor = dark ? "text-white" : "text-[#1d1d1f]";
  const subTextColor = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const cardBg = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";

  // 곡 목록 로드
  useEffect(() => {
    axios.get(`${BASE_URL}/songs`).then((r) => setSongs(r.data));
  }, []);

  // 검색 페이지에서 곡을 선택해 넘어온 경우 자동 선택
  useEffect(() => {
    if (!songs.length || selectedSong) return;
    if (!initialSongId && !initialSongTitle) return;
    const normalizedId = Number(initialSongId);
    const matched =
      songs.find((item) => Number.isFinite(normalizedId) && item.song_id === normalizedId) ||
      songs.find(
        (item) =>
          !!initialSongTitle &&
          item.title === initialSongTitle &&
          (!initialSongArtist || item.artist === initialSongArtist),
      );
    if (matched) {
      setSelectedSong(matched);
      setSearchQuery("");
      setShowList(false);
      setSemitones(0);
    }
  }, [songs, selectedSong, initialSongId, initialSongTitle, initialSongArtist]);

  // 곡 선택 시 피치 프레임 로드
  useEffect(() => {
    if (!selectedSong) return;
    axios
      .get(`${BASE_URL}/songs/${selectedSong.song_id}/pitch-frames`)
      .then((r) => setPitchFrames(r.data))
      .catch(() => setPitchFrames(null));
  }, [selectedSong]);

  // 마이크 상태 변경 시
  useEffect(() => {
    if (!isMicOn) {
      stopMic();
      return;
    }
    startMic();
    return () => stopMic();
  }, [isMicOn]);

  // semitones ref 동기화
  useEffect(() => { semitonesRef.current = semitones; }, [semitones]);

  // pitchFrames ref 동기화 (sampleInterval 클로저에서 사용)
  useEffect(() => { pitchFramesRef.current = pitchFrames; }, [pitchFrames]);

  // selectedSong ref 동기화 (drawCanvas 클로저에서 사용)
  useEffect(() => { selectedSongRef.current = selectedSong; }, [selectedSong]);

  // Canvas RAF: 그리기만 담당 (항상 60fps)
  useEffect(() => {
    function loop() {
      drawCanvas();
      canvasRafRef.current = requestAnimationFrame(loop);
    }
    canvasRafRef.current = requestAnimationFrame(loop);
    return () => { if (canvasRafRef.current) cancelAnimationFrame(canvasRafRef.current); };
  }, [isDarkMode]);

  async function startMic() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    micStreamRef.current = stream;
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    audioCtxRef.current = ctx;

    const mic = ctx.createMediaStreamSource(stream);

    // 피치 분석용 analyser
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    mic.connect(analyser);
    analyserRef.current = analyser;

    // 합성 임펄스 응답으로 리버브 생성
    const irLen = Math.floor(ctx.sampleRate * 4);
    const ir = ctx.createBuffer(2, irLen, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = ir.getChannelData(ch);
      for (let i = 0; i < irLen; i++)
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / irLen, 4);
    }
    const convolver = ctx.createConvolver();
    convolver.buffer = ir;

    const dryGain = ctx.createGain();
    dryGain.gain.value = 0.78;
    const reverbGain = ctx.createGain();
    reverbGain.gain.value = 0.22;

    // 모니터링: mic → wetGain → [dry + reverb] → panner(오른쪽) → destination
    const wetGain = ctx.createGain();
    wetGain.gain.value = 0;
    echoGainRef.current = wetGain;

    const micPanner = ctx.createStereoPanner();
    micPanner.pan.value = 1; // 오른쪽

    mic.connect(wetGain);
    wetGain.connect(dryGain);
    wetGain.connect(convolver);
    dryGain.connect(micPanner);
    convolver.connect(reverbGain);
    reverbGain.connect(micPanner);
    micPanner.connect(ctx.destination);

    // 통합 RAF 루프에서 사용할 detector/buf를 ref에 저장
    detectorRef.current = PitchDetector.forFloat32Array(analyser.fftSize);
    micBufRef.current = new Float32Array(analyser.fftSize);
  }

  function toggleEcho() {
    const gain = echoGainRef.current;
    if (!gain) return;
    const next = !isEchoOn;
    gain.gain.value = next ? 0.8 : 0;
    setIsEchoOn(next);
  }

  function stopMic() {
    detectorRef.current = null;
    micBufRef.current = null;
    if (echoGainRef.current) {
      echoGainRef.current.gain.value = 0;
      echoGainRef.current.disconnect();
      echoGainRef.current = null;
    }
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    analyserRef.current = null;
    setIsEchoOn(false);
    setUserPitch(null);
  }

  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;

    // 레이아웃
    const LABEL_W = 44; // 왼쪽 노트 레이블 영역
    const PLOT_W = W - LABEL_W;

    // 선택된 곡의 음역대 기준으로 동적 범위 계산 (없으면 C3~C5 기본값)
    const song = selectedSongRef.current;
    const semi = semitonesRef.current;
    const PAD = 5; // 위아래 패딩 (반음)
    const rawMin = song ? song.midi_min + semi : 48;
    const rawMax = song ? song.midi_max + semi : 72;
    // C 경계로 맞춤 (더 깔끔한 그리드)
    const MIDI_MIN = Math.max(24, Math.floor((rawMin - PAD) / 12) * 12);
    const MIDI_MAX = Math.min(96, Math.ceil((rawMax + PAD) / 12) * 12);
    const MIDI_CENTER = 60; // C4 고정

    function midiToY(m: number) {
      return H - ((m - MIDI_MIN) / (MIDI_MAX - MIDI_MIN)) * H;
    }

    // 배경
    ctx.fillStyle = isDarkMode ? "#0d0d0d" : "#f0f0f0";
    ctx.fillRect(0, 0, W, H);

    // 레이블 영역 배경
    ctx.fillStyle = isDarkMode ? "#161616" : "#e8e8e8";
    ctx.fillRect(0, 0, LABEL_W, H);

    // 구분선
    ctx.strokeStyle = isDarkMode ? "#2a2a2a" : "#ccc";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LABEL_W, 0);
    ctx.lineTo(LABEL_W, H);
    ctx.stroke();

    // 그리드 & 레이블
    for (let m = MIDI_MIN; m <= MIDI_MAX; m++) {
      const isC = m % 12 === 0;
      const isBlack = [1, 3, 6, 8, 10].includes(m % 12);
      const y = midiToY(m);

      // 플롯 영역 수평선
      if (isC) {
        ctx.strokeStyle = isDarkMode ? "#333" : "#c0c0c0";
        ctx.lineWidth = 1.2;
      } else if (!isBlack) {
        ctx.strokeStyle = isDarkMode ? "#1a1a1a" : "#e0e0e0";
        ctx.lineWidth = 0.5;
      } else {
        continue; // 검은 건반 위치는 선 생략
      }
      ctx.beginPath();
      ctx.moveTo(LABEL_W, y);
      ctx.lineTo(W, y);
      ctx.stroke();

      // C4 강조선
      if (m === MIDI_CENTER) {
        ctx.strokeStyle = isDarkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)";
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(LABEL_W, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      // 노트 레이블 (C만)
      if (isC) {
        ctx.fillStyle = m === MIDI_CENTER
          ? "#00d9b1"
          : (isDarkMode ? "#555" : "#999");
        ctx.font = `${m === MIDI_CENTER ? "bold" : ""} 11px monospace`;
        ctx.textAlign = "center";
        ctx.fillText(midiToNoteName(m), LABEL_W / 2, y + 4);
      }
    }

    const step = PLOT_W / MAX_HISTORY;
    const CENTER_X = LABEL_W + PLOT_W / 2;

    // 현재 위치 기준선
    ctx.strokeStyle = isDarkMode ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.12)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(CENTER_X, 0);
    ctx.lineTo(CENTER_X, H);
    ctx.stroke();
    ctx.setLineDash([]);

    function drawTrack(
      history: (number | null)[],
      lineColor: string,
      dotColor: string,
      lineWidth: number,
      fillNull: number, // null일 때 대체 MIDI (C4)
      isReal: boolean,  // 실제 피치 있을 때만 점
    ) {
      const N = history.length;
      function xOf(i: number) { return CENTER_X - (N - 1 - i) * step; }

      ctx.strokeStyle = lineColor;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      let started = false;
      history.forEach((m, i) => {
        const val = (m === null || m < MIDI_MIN || m > MIDI_MAX) ? fillNull : m;
        const x = xOf(i), y = midiToY(val);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // 실제 피치 구간만 캡슐로 강조
      if (isReal) {
        const capW = step * 1;   // 가로: 샘플 간격의 3배
        const capH = 7;          // 세로: 고정 높이
        const r = capH / 2;
        ctx.fillStyle = dotColor;
        history.forEach((m, i) => {
          if (m === null || m < MIDI_MIN || m > MIDI_MAX) return;
          const cx = xOf(i), cy = midiToY(m);
          ctx.beginPath();
          ctx.roundRect(cx - capW / 2, cy - r, capW, capH, r);
          ctx.fill();
        });
      }
    }

    // 원곡 피치 — pitchFrames 기반 단일 연속선 (과거+미래 통합)
    const pf = pitchFramesRef.current;
    const audio = audioRef.current;
    if (pf && audio) {
      const currentMs = audio.currentTime * 1000;
      const pixelsPerMs = step / 100;
      const halfMs = (PLOT_W / 2) / pixelsPerMs;
      const startStep = Math.ceil((currentMs - halfMs) / 100);
      const endStep = Math.floor((currentMs + halfMs) / 100);

      // 100ms 간격 키포인트 수집 (null → MIDI_CENTER)
      type KP = { x: number; y: number; real: boolean };
      const kps: KP[] = [];
      for (let si = startStep; si <= endStep; si++) {
        const frameMs = si * 100;
        const fi = Math.round(frameMs / pf.hop_ms);
        const x = CENTER_X + (frameMs - currentMs) * pixelsPerMs;
        let midi = MIDI_CENTER;
        let real = false;
        if (fi >= 0 && fi < pf.hz.length) {
          const hz = pf.hz[fi];
          const raw = hz ? hzToMidi(hz as number) + semi : null;
          if (raw !== null && raw >= MIDI_MIN && raw <= MIDI_MAX) { midi = raw; real = true; }
        }
        kps.push({ x, y: midiToY(midi), real });
      }

      // Catmull-Rom 스플라인 — 각 구간을 bezier로 변환해 부드럽게 연결
      ctx.strokeStyle = "#00c49a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (kps.length > 0) {
        ctx.moveTo(kps[0].x, kps[0].y);
        for (let i = 0; i < kps.length - 1; i++) {
          const p0 = kps[Math.max(0, i - 1)];
          const p1 = kps[i];
          const p2 = kps[i + 1];
          const p3 = kps[Math.min(kps.length - 1, i + 2)];
          ctx.bezierCurveTo(
            p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6,
            p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6,
            p2.x, p2.y,
          );
        }
      }
      ctx.stroke();

      // 캡슐 — 실제 피치 있는 키포인트만
      const capW = step;
      const capH = 7;
      const capR = capH / 2;
      ctx.fillStyle = "#00ffce";
      kps.forEach(({ x, y, real }) => {
        if (!real) return;
        ctx.beginPath();
        ctx.roundRect(x - capW / 2, y - capR, capW, capH, capR);
        ctx.fill();
      });
    } else {
      const shiftedOrig = origPitchHistory.current.map((m) => m !== null ? m + semi : null);
      drawTrack(shiftedOrig, "#00c49a", "#00ffce", 2, MIDI_CENTER, true);
    }

    // 사용자 피치 — RAF 60fps 실시간 읽기로 CENTER_X 싱크
    {
      let realtimeMidi: number | null = null;
      if (analyserRef.current && detectorRef.current && micBufRef.current && audioCtxRef.current) {
        const buf = micBufRef.current as Float32Array<ArrayBuffer>;
        analyserRef.current.getFloatTimeDomainData(buf);
        const [freq, clarity] = detectorRef.current.findPitch(buf, audioCtxRef.current.sampleRate);
        realtimeMidi = clarity > 0.8 && freq > 60 ? hzToMidi(freq) : null;
      }

      const currentMs2 = (audioRef.current?.currentTime ?? 0) * 1000;
      const pixelsPerMs2 = step / 100;
      // inputLatency + 버퍼 절반(fftSize/2) 만큼 시각 보정 (감지된 피치를 실제 발생 시점으로 당김)
      const latencyCompMs = (((audioCtxRef.current as any)?.inputLatency ?? 0.02) * 1000)
        + ((analyserRef.current?.fftSize ?? 1024) / 2 / (audioCtxRef.current?.sampleRate ?? 44100)) * 1000;
      type UKP = { x: number; y: number; real: boolean };
      const ukps: UKP[] = [];
      // 히스토리: 오디오 시간 기반 + 레이턴시 보정
      userPitchHistory.current.forEach(({ time, midi: m }) => {
        const x = CENTER_X + (time * 1000 - currentMs2 + latencyCompMs) * pixelsPerMs2;
        if (x > CENTER_X) return;
        const midi = (m === null || m < MIDI_MIN || m > MIDI_MAX) ? MIDI_CENTER : m;
        ukps.push({ x, y: midiToY(midi), real: m !== null && m >= MIDI_MIN && m <= MIDI_MAX });
      });
      // 실시간 현재 피치 — 레이턴시만큼 오른쪽으로 보정, CENTER_X 넘지 않게 클램프
      const rtX = Math.min(CENTER_X, CENTER_X + latencyCompMs * pixelsPerMs2);
      const rtMidi = (realtimeMidi === null || realtimeMidi < MIDI_MIN || realtimeMidi > MIDI_MAX) ? MIDI_CENTER : realtimeMidi;
      ukps.push({ x: rtX, y: midiToY(rtMidi), real: realtimeMidi !== null && realtimeMidi >= MIDI_MIN && realtimeMidi <= MIDI_MAX });

      // Catmull-Rom 스플라인으로 부드럽게 연결
      ctx.strokeStyle = isDarkMode ? "rgba(255,255,255,0.75)" : "#5a7df5";
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (ukps.length > 0) {
        ctx.moveTo(ukps[0].x, ukps[0].y);
        for (let i = 0; i < ukps.length - 1; i++) {
          const p0 = ukps[Math.max(0, i - 1)];
          const p1 = ukps[i];
          const p2 = ukps[i + 1];
          const p3 = ukps[Math.min(ukps.length - 1, i + 2)];
          ctx.bezierCurveTo(
            p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6,
            p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6,
            p2.x, p2.y,
          );
        }
      }
      ctx.stroke();

      // 캡슐 — 실제 피치 있는 위치만
      const userDotColor = isDarkMode ? "#fff" : "#4c6ef5";
      const capW = step;
      const capH = 7;
      const capR = capH / 2;
      ctx.fillStyle = userDotColor;
      ukps.forEach(({ x, y, real }) => {
        if (!real) return;
        ctx.beginPath();
        ctx.roundRect(x - capW / 2, y - capR, capW, capH, capR);
        ctx.fill();
      });
    }
  }

  function stopSampling() {
    if (sampleIntervalRef.current) {
      clearInterval(sampleIntervalRef.current);
      sampleIntervalRef.current = null;
    }
  }

  async function playWithSemitones(song: Song, semi: number, model: "kim" | "mdx" = "kim") {
    stopSampling();
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.crossOrigin = 'anonymous'; // WebAudio CORS 필요
    } else {
      audioRef.current.pause();
    }

    audioRef.current.src = `${BASE_URL}/songs/${song.song_id}/accompaniment?semitones=${semi}&model=${model}`;
    audioRef.current.onended = () => { stopSampling(); setIsPlaying(false); setSongEnded(true); };

    // 반주를 WebAudio로 라우팅해 왼쪽 채널로 출력 (최초 1회만 연결)
    if (!accompCtxRef.current) {
      const accompCtx = new AudioContext({ latencyHint: 'interactive' });
      accompCtxRef.current = accompCtx;
      const source = accompCtx.createMediaElementSource(audioRef.current);
      accompSourceRef.current = source;
      const panner = accompCtx.createStereoPanner();
      panner.pan.value = -1; // 왼쪽
      source.connect(panner);
      panner.connect(accompCtx.destination);
    }
    await accompCtxRef.current.resume();

    await audioRef.current.play();

    // 재생 시작 순간 히스토리/로그 초기화 후 interval 시작
    userPitchHistory.current = [];
    origPitchHistory.current = [];
    fullSongLog.current = [];
    setSongEnded(false);
    setShowAnalysis(false);
    setIsPlaying(true);

    sampleIntervalRef.current = setInterval(() => {
      const pf = pitchFramesRef.current;
      const audio = audioRef.current;
      let origMidi: number | null = null;
      let userMidi: number | null = null;

      // 원곡 피치
      if (pf && audio) {
        const idx = Math.round((audio.currentTime * 1000) / pf.hop_ms);
        const hz = idx < pf.hz.length ? pf.hz[idx] : null;
        origMidi = hz ? hzToMidi(hz as number) : null;
        setOriginalPitch(origMidi);
        origPitchHistory.current.push(origMidi);
        if (origPitchHistory.current.length > MAX_HISTORY) origPitchHistory.current.shift();
      }

      // 마이크 피치
      if (analyserRef.current && audioCtxRef.current && detectorRef.current && micBufRef.current) {
        const buf = micBufRef.current as Float32Array<ArrayBuffer>;
        analyserRef.current.getFloatTimeDomainData(buf);
        const [freq, clarity] = detectorRef.current.findPitch(buf, audioCtxRef.current.sampleRate);
        userMidi = clarity > 0.8 && freq > 60 ? hzToMidi(freq) : null;
        setUserPitch(userMidi);
        userPitchHistory.current.push({ time: audio?.currentTime ?? 0, midi: userMidi });
        if (userPitchHistory.current.length > MAX_HISTORY) userPitchHistory.current.shift();
      }

      // 전체 분석 로그 (마이크 켜져 있을 때만 의미 있음)
      if (audio) fullSongLog.current.push({ time: audio.currentTime, userMidi, origMidi });
    }, 100);
  }

  async function togglePlay() {
    if (!selectedSong) return;
    if (isPlaying) {
      audioRef.current?.pause();
      stopSampling();
      setIsPlaying(false);
      return;
    }
    await playWithSemitones(selectedSong, semitones, accompanimentModel);
  }

  async function changeSemitones(val: number) {
    setSemitones(val);
    if (isPlaying && selectedSong) {
      await playWithSemitones(selectedSong, val, accompanimentModel);
    }
  }

  async function changeModel(model: "kim" | "mdx") {
    setAccompanimentModel(model);
    if (isPlaying && selectedSong) {
      await playWithSemitones(selectedSong, semitones, model);
    }
  }

  function calcAnalysis() {
    const log = fullSongLog.current;
    const withOrig = log.filter(d => d.origMidi !== null);
    const withBoth = withOrig.filter(d => d.userMidi !== null);
    if (withOrig.length === 0 || withBoth.length === 0) return null;

    const semi = semitonesRef.current;
    let perfect = 0, close = 0, ok = 0, off = 0;
    withBoth.forEach(({ userMidi, origMidi }) => {
      const diff = Math.abs(userMidi! - (origMidi! + semi));
      if (diff <= 1) perfect++;
      else if (diff <= 2.5) close++;
      else if (diff <= 5) ok++;
      else off++;
    });
    const n = withBoth.length;
    const accuracy = (perfect + close * 0.7 + ok * 0.4) / n;
    const coverage = Math.min(n / withOrig.length, 1);
    const score = Math.round((accuracy * 0.7 + coverage * 0.3) * 100);
    const avgDiff = withBoth.reduce((s, d) => s + Math.abs(d.userMidi! - (d.origMidi! + semi)), 0) / n;
    return {
      score,
      coverage: Math.round(coverage * 100),
      perfect: Math.round(perfect / n * 100),
      close: Math.round(close / n * 100),
      ok: Math.round(ok / n * 100),
      off: Math.round(off / n * 100),
      avgDiff: Math.round(avgDiff * 10) / 10,
    };
  }

  function scoreLabel(s: number) {
    if (s >= 90) return { text: "완벽해요!", color: "#00d9b1" };
    if (s >= 78) return { text: "아주 잘 하셨어요!", color: "#00d9b1" };
    if (s >= 65) return { text: "잘 하셨어요!", color: "#60a5fa" };
    if (s >= 50) return { text: "나쁘지 않아요", color: "#fbbf24" };
    return { text: "더 연습해봐요", color: "#f87171" };
  }

  // 사용자 추천 키
  const recommendedKey = selectedSong && user?.midi_median
    ? Math.round(user.midi_median - selectedSong.midi_median)
    : null;

  const diff = userPitch !== null && originalPitch !== null
    ? Math.round((userPitch - (originalPitch + semitones)) * 10) / 10
    : null;

  return (
    <div className={`min-h-screen ${dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]"} relative overflow-hidden`}>
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] rounded-full blur-[120px] opacity-20 bg-[#00d9b1]" />
        <div className={`absolute bottom-0 right-0 w-[400px] h-[400px] rounded-full blur-[100px] opacity-10 ${dark ? "bg-blue-500" : "bg-blue-400"}`} />
      </div>

      <div className="relative z-10 min-h-screen flex flex-col font-['Pretendard']">
        <header className={`fixed top-0 left-0 right-0 z-50 ${dark ? "bg-[#0a0a0a]/80" : "bg-[#f5f5f7]/80"} backdrop-blur-xl border-b ${border}`}>
          <div className="max-w-6xl mx-auto px-8 h-16 flex items-center justify-between">
            <button onClick={onBack} className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${cardBg} border ${border} ${dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]"}`}>
              <ArrowLeft className={`w-4 h-4 ${subTextColor}`} />
            </button>
            <span className={`text-[15px] font-semibold tracking-tight ${textColor}`}>PitchWizard</span>
            <div className="w-8" />
          </div>
        </header>

        <main className="flex-1 pt-16 px-8 pb-16 max-w-6xl mx-auto w-full">
          {/* 히어로 */}
          <div className="text-center pt-16 pb-10">
            <div className={`inline-flex items-center gap-2 text-[12px] px-3 py-1 rounded-full border ${border} ${cardBg} ${subTextColor} mb-7`}>
              <span className="w-1.5 h-1.5 rounded-full bg-[#00d9b1] animate-pulse" />
              Live Pitch
            </div>
            <h2 className={`text-[52px] font-bold leading-[1.05] tracking-tight ${textColor} mb-4`}>
              반주에 맞춰<br />
              <span className="text-[#00d9b1]">불러보세요</span>
            </h2>
            <p className={`text-[16px] ${subTextColor} leading-relaxed max-w-md mx-auto`}>
              실시간 피치 분석으로 정확도를 확인하세요
            </p>
          </div>
          <div className="space-y-4">

          {/* 곡 검색 */}
          <div className={`rounded-2xl border ${border} ${cardBg} p-6 backdrop-blur-xl`}>
            <p className={`text-sm font-medium mb-3 ${subTextColor}`}>곡 검색</p>
            <div className="relative">
              <div className={`flex items-center gap-3 rounded-xl border ${border} px-4 py-3 ${isDarkMode ? "bg-white/5" : "bg-black/5"}`}>
                <Search className={`w-4 h-4 flex-shrink-0 ${subTextColor}`} />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => { setSearchQuery(e.target.value); setShowList(true); }}
                  onFocus={() => setShowList(true)}
                  placeholder="제목 또는 가수 검색"
                  className={`flex-1 bg-transparent outline-none text-[15px] ${textColor} placeholder:${subTextColor}`}
                />
                {selectedSong && (
                  <button onClick={() => { setSelectedSong(null); setSearchQuery(""); setIsPlaying(false); }}>
                    <X className={`w-4 h-4 ${subTextColor}`} />
                  </button>
                )}
              </div>

              {/* 선택된 곡 표시 */}
              {selectedSong && (
                <div className="mt-2 px-1">
                  <span className="text-[#00d9b1] text-sm font-medium">
                    ✓ {selectedSong.title} - {selectedSong.artist}
                  </span>
                </div>
              )}

              {/* 검색 결과 드롭다운 */}
              {showList && searchQuery && (
                <div className={`absolute z-50 w-full mt-2 rounded-xl border ${border} overflow-hidden shadow-2xl ${isDarkMode ? "bg-[#1f1f1f]" : "bg-white"}`}>
                  {songs
                    .filter((s) =>
                      s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      s.artist.toLowerCase().includes(searchQuery.toLowerCase())
                    )
                    .slice(0, 8)
                    .map((s) => (
                      <button
                        key={s.song_id}
                        className={`w-full text-left px-4 py-3 text-[14px] ${textColor} transition-colors ${isDarkMode ? "hover:bg-white/10" : "hover:bg-black/5"} border-b last:border-b-0 ${border}`}
                        onClick={() => {
                          setSelectedSong(s);
                          setSearchQuery("");
                          setShowList(false);
                          setIsPlaying(false);
                          setSemitones(0);
                        }}
                      >
                        <span className="font-medium">{s.title}</span>
                        <span className={`ml-2 text-[13px] ${subTextColor}`}>{s.artist}</span>
                      </button>
                    ))}
                  {songs.filter((s) =>
                    s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                    s.artist.toLowerCase().includes(searchQuery.toLowerCase())
                  ).length === 0 && (
                    <div className={`px-4 py-4 text-sm text-center ${subTextColor}`}>검색 결과가 없습니다</div>
                  )}
                </div>
              )}
            </div>
          </div>

          {selectedSong && (
            <>
              {/* 키 조절 + 재생 */}
              <div className={`rounded-2xl border ${border} ${cardBg} p-6 backdrop-blur-xl`}>
                {/* 반주 모델 선택 */}
                <div className="flex items-center gap-3 mb-5">
                  <p className={`text-sm font-medium ${subTextColor}`}>반주 모델</p>
                  <div className={`flex rounded-lg border ${border} overflow-hidden text-xs`}>
                    <button
                      onClick={() => changeModel("kim")}
                      className={`px-3 py-1.5 transition-colors ${
                        accompanimentModel === "kim"
                          ? "bg-[#00d9b1] text-white font-semibold"
                          : `${cardBg} ${subTextColor}`
                      }`}
                    >
                      Kim Vocal
                    </button>
                    <button
                      onClick={() => changeModel("mdx")}
                      className={`px-3 py-1.5 transition-colors ${
                        accompanimentModel === "mdx"
                          ? "bg-[#00d9b1] text-white font-semibold"
                          : `${cardBg} ${subTextColor}`
                      }`}
                    >
                      MDX Main
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between mb-4">
                  <p className={`text-sm font-medium ${subTextColor}`}>키 조절</p>
                  {recommendedKey !== null && (
                    <button
                      onClick={() => changeSemitones(Math.max(-5, Math.min(5, recommendedKey)))}
                      className="text-xs px-3 py-1 rounded-full bg-[#00d9b1]/20 text-[#00d9b1] border border-[#00d9b1]/30"
                    >
                      추천 키 {recommendedKey > 0 ? `+${recommendedKey}` : recommendedKey} 적용
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-4">
                  <span className={`text-[28px] font-bold w-16 text-center ${textColor}`}>
                    {semitones > 0 ? `+${semitones}` : semitones}
                  </span>
                  <input
                    type="range" min={-5} max={5} step={1}
                    value={semitones}
                    onChange={(e) => changeSemitones(Number(e.target.value))}
                    className="flex-1 accent-[#00d9b1]"
                  />
                  <button
                    onClick={togglePlay}
                    className="w-14 h-14 rounded-full bg-[#00d9b1] flex items-center justify-center shadow-lg hover:opacity-90 transition"
                  >
                    {isPlaying ? <Square className="w-6 h-6 text-white" /> : <Play className="w-6 h-6 text-white" />}
                  </button>
                </div>
              </div>

              {/* 실시간 피치 Canvas */}
              <div className={`rounded-2xl border ${border} ${cardBg} p-6 backdrop-blur-xl`}>
                <div className="flex items-center justify-between mb-4">
                  <p className={`text-sm font-medium ${subTextColor}`}>실시간 피치 비교</p>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#00d9b1] inline-block" />원곡</span>
                    <span className={`flex items-center gap-1 ${isDarkMode ? "text-white" : "text-[#4c6ef5]"}`}>
                      <span className={`w-3 h-0.5 inline-block ${isDarkMode ? "bg-white" : "bg-[#4c6ef5]"}`} />내 목소리
                    </span>
                    <button
                      onClick={() => setIsMicOn((v) => !v)}
                      className={`flex items-center gap-1 px-3 py-1 rounded-full border transition ${
                        isMicOn
                          ? "bg-[#00d9b1]/20 border-[#00d9b1]/50 text-[#00d9b1]"
                          : `border-white/20 ${subTextColor}`
                      }`}
                    >
                      {isMicOn ? <Mic className="w-3 h-3" /> : <MicOff className="w-3 h-3" />}
                      {isMicOn ? "마이크 ON" : "마이크 OFF"}
                    </button>
                    {isMicOn && (
                      <button
                        onClick={toggleEcho}
                        className={`flex items-center gap-1 px-3 py-1 rounded-full border transition text-xs ${
                          isEchoOn
                            ? "bg-purple-500/20 border-purple-400/50 text-purple-300"
                            : `border-white/20 ${subTextColor}`
                        }`}
                      >
                        🎙 에코 {isEchoOn ? "ON" : "OFF"}
                      </button>
                    )}
                  </div>
                </div>

                <canvas
                  ref={canvasRef}
                  width={1200}
                  height={400}
                  className="w-full rounded-xl"
                  style={{ height: "280px" }}
                />

                {/* 현재 피치 수치 */}
                <div className="mt-4 flex gap-6 text-sm">
                  <div>
                    <p className={subTextColor}>원곡 피치</p>
                    <p className={`text-lg font-bold ${textColor}`}>
                      {originalPitch !== null ? midiToNoteName(Math.round(originalPitch + semitones)) : "--"}
                    </p>
                  </div>
                  <div>
                    <p className={subTextColor}>내 피치</p>
                    <p className={`text-lg font-bold ${textColor}`}>
                      {userPitch !== null ? midiToNoteName(Math.round(userPitch)) : "--"}
                    </p>
                  </div>
                  {diff !== null && (
                    <div>
                      <p className={subTextColor}>차이</p>
                      <p className={`text-lg font-bold ${Math.abs(diff) < 0.5 ? "text-[#00d9b1]" : "text-red-400"}`}>
                        {diff > 0 ? `+${diff}` : diff} 반음
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {!pitchFrames && (
                <p className={`text-sm text-center ${subTextColor}`}>
                  이 곡은 피치 프레임 데이터가 없습니다. 원곡 라인은 표시되지 않아요.
                </p>
              )}

              {/* 완곡 후 분석 버튼 */}
              {songEnded && !showAnalysis && (
                <div className="flex justify-center">
                  <button
                    onClick={() => setShowAnalysis(true)}
                    className="px-8 py-3 rounded-full bg-[#00d9b1] text-white font-semibold text-[15px] shadow-lg hover:opacity-90 transition"
                  >
                    피치 분석 결과 보기
                  </button>
                </div>
              )}

              {/* 분석 결과 패널 */}
              {showAnalysis && (() => {
                const r = calcAnalysis();
                if (!r) return (
                  <div className={`rounded-2xl border ${border} ${cardBg} p-6 text-center ${subTextColor} text-sm`}>
                    마이크가 꺼져 있었거나 데이터가 부족합니다.
                  </div>
                );
                const lbl = scoreLabel(r.score);
                const bars = [
                  { label: "정확", sub: "±1반음", pct: r.perfect, color: "#00d9b1" },
                  { label: "근접", sub: "±2.5반음", pct: r.close, color: "#60a5fa" },
                  { label: "보통", sub: "±5반음", pct: r.ok, color: "#fbbf24" },
                  { label: "이탈", sub: ">5반음", pct: r.off, color: "#f87171" },
                ];
                return (
                  <div className={`rounded-2xl border ${border} ${cardBg} p-6 backdrop-blur-xl space-y-6`}>
                    <div className="flex items-center justify-between">
                      <p className={`text-sm font-medium ${subTextColor}`}>피치 분석 결과</p>
                      <button onClick={() => setShowAnalysis(false)} className={`text-xs ${subTextColor} hover:opacity-70`}>닫기</button>
                    </div>

                    {/* 점수 */}
                    <div className="flex items-center gap-6">
                      <div className="relative w-28 h-28 flex-shrink-0">
                        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                          <circle cx="50" cy="50" r="42" fill="none" stroke={isDarkMode ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"} strokeWidth="10" />
                          <circle cx="50" cy="50" r="42" fill="none" stroke={lbl.color} strokeWidth="10"
                            strokeDasharray={`${2 * Math.PI * 42 * r.score / 100} ${2 * Math.PI * 42}`}
                            strokeLinecap="round" />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className={`text-2xl font-bold ${textColor}`}>{r.score}</span>
                          <span className={`text-[10px] ${subTextColor}`}>점</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-[22px] font-bold" style={{ color: lbl.color }}>{lbl.text}</p>
                        <p className={`text-sm mt-1 ${subTextColor}`}>
                          노래한 구간 <span className={textColor}>{r.coverage}%</span>
                          &nbsp;·&nbsp;평균 이탈 <span className={textColor}>{r.avgDiff} 반음</span>
                        </p>
                      </div>
                    </div>

                    {/* 피치 정확도 바 */}
                    <div className="space-y-2">
                      {bars.map(b => (
                        <div key={b.label} className="flex items-center gap-3">
                          <div className="w-16 text-right">
                            <span className={`text-xs font-medium ${textColor}`}>{b.label}</span>
                            <span className={`block text-[10px] ${subTextColor}`}>{b.sub}</span>
                          </div>
                          <div className={`flex-1 h-4 rounded-full overflow-hidden ${isDarkMode ? "bg-white/10" : "bg-black/8"}`}>
                            <div
                              className="h-full rounded-full transition-all"
                              style={{ width: `${b.pct}%`, backgroundColor: b.color }}
                            />
                          </div>
                          <span className={`text-xs w-9 text-right ${textColor}`}>{b.pct}%</span>
                        </div>
                      ))}
                    </div>

                    <button
                      onClick={() => { setSongEnded(false); setShowAnalysis(false); }}
                      className={`w-full py-2.5 rounded-xl border ${border} text-sm ${subTextColor} hover:opacity-70 transition`}
                    >
                      닫고 다시 부르기
                    </button>
                  </div>
                );
              })()}
            </>
          )}
          </div>
        </main>
      </div>
    </div>
  );
}
