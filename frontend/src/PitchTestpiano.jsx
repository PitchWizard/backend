// PitchTestpiano.jsx
import React, { useEffect, useRef, useState } from "react";
import { Mic, Square, Download } from "lucide-react";

const DEFAULTS = {
  measureWindowSec: 2.5,
  voiceOnsetRmsThreshold: 0.015,
  frameIntervalMs: 60,
  strongCents: 40,
  weakCents: 75,
  strongPercent: 0.6,
  weakPercent: 0.4,
};

function midiToFreq(m) {
  return 440 * Math.pow(2, (m - 69) / 12);
}
function freqToMidi(freq) {
  return 69 + 12 * Math.log2(freq / 440);
}
function freqToCents(fTarget, fMeasured) {
  if (fMeasured <= 0 || fTarget <= 0) return Infinity;
  return 1200 * Math.log2(fMeasured / fTarget);
}
function midiToNoteName(m) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return names[m % 12] + Math.floor(m / 12 - 1);
}

function generateNoteList() {
  const midiRange = [48, 84]; // C3~C6
  let list = [];
  for (let m = midiRange[0]; m <= midiRange[1]; m++) {
    const name = midiToNoteName(m);
    if (!name.includes("#")) list.push({ note: name, midi: m, freq: midiToFreq(m) });
  }
  return list;
}
const NOTES_TO_TEST = generateNoteList();

function autocorrelate(buffer, sampleRate) {
  const SIZE = buffer.length;
  let rms = 0;
  for (let i = 0; i < SIZE; i++) rms += buffer[i] * buffer[i];
  rms = Math.sqrt(rms / SIZE);
  if (rms < 0.001) return { freq: -1, rms };

  let r = new Array(SIZE).fill(0);
  for (let lag = 0; lag < SIZE; lag++) {
    let sum = 0;
    for (let i = 0; i < SIZE - lag; i++) sum += buffer[i] * buffer[i + lag];
    r[lag] = sum;
  }

  let d = 0;
  while (d < SIZE && r[d] > r[d + 1]) d++;
  let maxPos = -1,
    maxVal = -Infinity;
  for (let i = d; i < SIZE; i++) {
    if (r[i] > maxVal) {
      maxVal = r[i];
      maxPos = i;
    }
  }
  if (maxVal <= 0 || maxPos === -1) return { freq: -1, rms };

  const left = r[maxPos - 1] ?? 0;
  const center = r[maxPos];
  const right = r[maxPos + 1] ?? 0;
  const denom = left - 2 * center + right;
  let shift = 0;
  if (denom !== 0) shift = (left - right) / (2 * denom);
  const lag = maxPos + shift;
  const freq = sampleRate / lag;
  return { freq, rms };
}

// ===== WAV 인코딩 =====
function interleave(buffers, totalLen) {
  const result = new Float32Array(totalLen);
  let offset = 0;
  for (const b of buffers) {
    result.set(b, offset);
    offset += b.length;
  }
  return result;
}
function floatTo16BitPCM(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view;
}
function encodeWAV(float32Array, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + float32Array.length * bytesPerSample);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + float32Array.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, float32Array.length * bytesPerSample, true);
  const pcm = floatTo16BitPCM(float32Array);
  for (let i = 0; i < pcm.byteLength; i++) view.setUint8(44 + i, pcm.getUint8(i));
  return new Blob([view], { type: "audio/wav" });
}

// 테시투라 계산 함수
function estimateTessitura(results, opts = {}) {
  const { strongThreshold = 0.6, minNotes = 3, maxAllowedGaps = 1 } = opts;
  const strongMask = results.map((r) => r.strong >= strongThreshold);
  const segments = [];
  let i = 0;

  while (i < results.length) {
    if (!strongMask[i]) {
      i++;
      continue;
    }
    let start = i;
    let end = i;
    let gaps = 0;
    i++;
    while (i < results.length) {
      if (strongMask[i]) {
        end = i;
        gaps = 0;
      } else {
        gaps++;
        if (gaps > maxAllowedGaps) break;
      }
      i++;
    }
    const included = [];
    for (let k = start; k <= end; k++) if (strongMask[k]) included.push(k);
    if (included.length >= minNotes) {
      const idxLow = included[0];
      const idxHigh = included[included.length - 1];
      const notes = included.map((idx) => results[idx].note);
      const avgStrong = included.reduce((s, idx) => s + results[idx].strong, 0) / included.length;
      segments.push({
        low: results[idxLow].note,
        high: results[idxHigh].note,
        notes,
        length: included.length,
        avgStrong,
      });
    }
  }
  if (segments.length === 0) return { tessitura: null, segments };
  segments.sort((a, b) => b.length - a.length || b.avgStrong - a.avgStrong);
  return { tessitura: segments[0], segments };
}

import saveVocalRange from "./api/vocalRangeApi";

export default function PitchTestPiano({ userId, onTestComplete }) {
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const canvasRef = useRef(null);

  const recBuffers = useRef([]);
  const recLength = useRef(0);
  const isRecording = useRef(false);

  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState([]);
  const [currentNote, setCurrentNote] = useState(null);
  const [pitchHistory, setPitchHistory] = useState([]);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [tessitura, setTessitura] = useState(null);
  const [retriedNotes, setRetriedNotes] = useState([]);
  const [retryingNote, setRetryingNote] = useState(null);

  useEffect(() => () => stopAll(), []);
  useEffect(() => drawCanvas(), [pitchHistory, currentNote]);

  async function initAudio() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const mic = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    mic.connect(analyser);
    analyserRef.current = analyser;

    const node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => {
      if (!isRecording.current) return;
      const input = e.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      recBuffers.current.push(copy);
      recLength.current += copy.length;
    };
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;
    node.connect(silentGain);
    silentGain.connect(ctx.destination);
    mic.connect(node);
  }

  function startRecording() {
    recBuffers.current = [];
    recLength.current = 0;
    isRecording.current = true;
    setDownloadUrl(null);
  }

  function stopRecording() {
    if (!isRecording.current) return;
    isRecording.current = false;
    const ctx = audioCtxRef.current;
    const data = interleave(recBuffers.current, recLength.current);
    const wavBlob = encodeWAV(data, ctx.sampleRate);
    const url = URL.createObjectURL(wavBlob);
    setDownloadUrl(url);

    recBuffers.current = [];
    recLength.current = 0;
  }

  function playTone(freq, duration = 1.2) {
    const ctx = audioCtxRef.current;
    const osc = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc2.type = "triangle";
    const mixGain = ctx.createGain();
    mixGain.gain.value = 0.6;
    osc.connect(mixGain);
    osc2.connect(mixGain);
    mixGain.connect(gain);
    gain.connect(ctx.destination);

    const now = ctx.currentTime;
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.7, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.4, now + 0.4);
    gain.gain.linearRampToValueAtTime(0.0001, now + duration);
    osc.frequency.value = freq;
    osc2.frequency.value = freq;
    osc.start();
    osc2.start();
    osc.stop(now + duration);
    osc2.stop(now + duration);
  }

  async function runNoteTest(noteObj) {
    setCurrentNote(noteObj.note);
    setPitchHistory([]);
    const analyser = analyserRef.current;
    const bufferLen = analyser.fftSize;
    const timeDomain = new Float32Array(bufferLen);
    const ctx = audioCtxRef.current;

    await new Promise((r) => setTimeout(r, 800));
    playTone(noteObj.freq, 1.2);
    await new Promise((r) => setTimeout(r, 1400));

    const frames = [];
    const history = [];
    const startTime = ctx.currentTime;
    while (ctx.currentTime - startTime < DEFAULTS.measureWindowSec) {
      analyser.getFloatTimeDomainData(timeDomain);
      const { freq } = autocorrelate(timeDomain, ctx.sampleRate);

      if (freq > 0) {
        const cents = Math.abs(freqToCents(noteObj.freq, freq));
        frames.push(cents);
        history.push(freq);
        setPitchHistory([...history]);
      }
      await new Promise((r) => setTimeout(r, DEFAULTS.frameIntervalMs));
    }

    const total = frames.length || 1;
    const strong = frames.filter((c) => c <= DEFAULTS.strongCents).length / total;
    const weak = frames.filter((c) => c <= DEFAULTS.weakCents).length / total;
    let grade = "Fail";
    if (strong >= DEFAULTS.strongPercent) grade = "Strong OK";
    else if (weak >= DEFAULTS.weakPercent) grade = "Weak OK";
    return { note: noteObj.note, strong, weak, grade };
  }

  async function startSequence() {
    setResults([]);
    setStatus("running");
    setRetriedNotes([]);
    setRetryingNote(null);
    await initAudio();
    startRecording();

    const res = [];
    let consecutiveFailCount = 0;
    for (const n of NOTES_TO_TEST) {
      const r = await runNoteTest(n);
      res.push(r);
      setResults([...res]);

      if (r.grade === "Fail") consecutiveFailCount += 1;
      else consecutiveFailCount = 0;

      if (consecutiveFailCount >= 3) {
        console.warn("❌ 연속 3회 실패 → 테스트 종료");
        break;
      }

      await new Promise((s) => setTimeout(s, 400));
    }

    stopRecording();

    const { tessitura, segments } = estimateTessitura(res, {
      strongThreshold: DEFAULTS.strongPercent,
      minNotes: 3,
      maxAllowedGaps: 1,
    });
    setTessitura(tessitura);
    console.log("🎼 Tessitura 분석 결과:", tessitura);
    console.log("📊 모든 구간:", segments);

    let midi_min = null, midi_max = null, midi_median = null;
    if (tessitura) {
      const midiValues = tessitura.notes.map(
        (n) => NOTES_TO_TEST.find((x) => x.note === n).midi
      );
      midiValues.sort((a, b) => a - b);
      midi_min = midiValues[0];
      midi_max = midiValues[midiValues.length - 1];
      midi_median =
        midiValues.length % 2 === 1
          ? midiValues[Math.floor(midiValues.length / 2)]
          : (midiValues[midiValues.length / 2 - 1] +
              midiValues[midiValues.length / 2]) /
            2;
    }

    if (tessitura && userId) {
      const payload = {
        user_id: userId,
        midi_min,
        midi_median,
        midi_max,
        low_note: tessitura.low,
        high_note: tessitura.high,
        avg_rms: null,
      };
      try {
        await saveVocalRange(payload);
        onTestComplete?.({ midi_min, midi_median, midi_max, low_note: tessitura.low, high_note: tessitura.high });
      } catch (e) {
        console.error("음역대 저장 실패:", e);
      }
    }

    setStatus("done");
  }

  async function retryNote(noteName) {
    if (retriedNotes.includes(noteName)) {
      alert(`${noteName} 음은 이미 재도전했습니다.`);
      return;
    }

    const noteObj = NOTES_TO_TEST.find((n) => n.note === noteName);
    if (!noteObj) return;

    const cur = results.find((r) => r.note === noteName);
    if (!cur || (cur.grade !== "Weak OK" && cur.grade !== "Fail")) {
      alert("재도전은 Weak OK 또는 Fail인 음만 가능합니다.");
      return;
    }

    const successGrades = ["Strong OK", "Weak OK"];
    const successIndices = results
      .map((r, i) => ({ i, grade: r.grade }))
      .filter((x) => successGrades.includes(x.grade))
      .map((x) => x.i);

    if (successIndices.length === 0) {
      alert("성공한 음이 없어서 재도전 대상 범위를 계산할 수 없습니다.");
      return;
    }

    setRetryingNote(noteName);
    setStatus("retrying");

    try {
      await initAudio();
      startRecording();

      const updated = await runNoteTest(noteObj);

      stopRecording();
      stopAll();

      setResults((prev) => {
        const next = prev.map((r) => (r.note === noteName ? updated : r));
        const { tessitura: newTessitura } = estimateTessitura(next, {
          strongThreshold: DEFAULTS.strongPercent,
          minNotes: 3,
          maxAllowedGaps: 1,
        });
        setTessitura(newTessitura);
        return next;
      });

      setRetriedNotes((prev) => [...prev, noteName]);
      setStatus("done");
    } catch (err) {
      console.error("retryNote error", err);
      alert("재도전 중 오류가 발생했습니다.");
      setStatus("done");
    } finally {
      setRetryingNote(null);
    }
  }

  function stopAll() {
    stopRecording();
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    setStatus("idle");
  }

  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width,
      height = canvas.height;
    const pad = { top: 26, right: 26, bottom: 24, left: 56 };
    const plotX = pad.left;
    const plotY = pad.top;
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    ctx.clearRect(0, 0, width, height);

    const minMidi = 36; // C2
    const maxMidi = 96; // C7

    const midiToY = (m) => plotY + ((maxMidi - m) / (maxMidi - minMidi)) * plotH;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(plotX, plotY, plotW, plotH);

    ctx.strokeStyle = "#d8dee8";
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, plotY, plotW, plotH);

    for (let m = minMidi; m <= maxMidi; m++) {
      const isOctave = m % 12 === 0;
      const y = midiToY(m);
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotW, y);
      ctx.strokeStyle = isOctave ? "#b9c2d0" : "#e8edf3";
      ctx.lineWidth = isOctave ? 1.5 : 1;
      ctx.stroke();

      if (isOctave) {
        ctx.fillStyle = "#334155";
        ctx.font = "600 13px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(midiToNoteName(m), plotX - 10, y);
      }
    }

    if (currentNote) {
      const noteObj = NOTES_TO_TEST.find((x) => x.note === currentNote);
      if (noteObj) {
        const y = midiToY(noteObj.midi);
        ctx.strokeStyle = "#ef476f";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([8, 8]);
        ctx.beginPath();
        ctx.moveTo(plotX, y);
        ctx.lineTo(plotX + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);

        const label = `목표 ${currentNote}`;
        ctx.font = "700 13px sans-serif";
        const labelW = ctx.measureText(label).width + 18;
        const labelH = 26;
        const labelX = plotX + plotW - labelW - 10;
        const labelY = Math.max(plotY + 8, Math.min(y - labelH - 8, plotY + plotH - labelH - 8));
        ctx.fillStyle = "#fff1f4";
        ctx.strokeStyle = "#ffc0cf";
        ctx.lineWidth = 1;
        roundRect(ctx, labelX, labelY, labelW, labelH, 8);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#c9184a";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, labelX + labelW / 2, labelY + labelH / 2);
      }
    }

    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    pitchHistory.forEach((f, i) => {
      const m = freqToMidi(f);
      const y = midiToY(m);
      const x = plotX + (i / Math.max(1, pitchHistory.length - 1)) * plotW;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineCap = "butt";
    ctx.lineJoin = "miter";
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  const isRunning = status === "running";
  const isBusy = status === "retrying";
  const statusLabel =
    status === "idle"
      ? "대기 중"
      : status === "running"
      ? "측정 중"
      : status === "retrying"
      ? "재측정 중"
      : "완료";

  return (
    <div className="relative p-6 md:p-8 text-white">
      <div className="grid gap-6 md:gap-7">
        <section className="rounded-3xl border border-white/10 bg-[linear-gradient(145deg,rgba(255,255,255,0.1),rgba(255,255,255,0.03))] shadow-[0_18px_45px_rgba(0,0,0,0.28)] backdrop-blur-xl overflow-hidden">
          <div className="px-5 md:px-7 pt-5 md:pt-6 pb-4 border-b border-white/10 flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm text-white/75">
              상태: <span className="text-white font-semibold">{statusLabel}</span>
              {currentNote && <span className="text-white/60"> · 현재 음: {currentNote}</span>}
            </div>
            <div className="text-xs text-white/55">측정 중에는 조용한 환경을 권장합니다</div>
          </div>

          <div className="p-5 md:p-7">
            <div className="overflow-hidden rounded-2xl border border-white/15 bg-white shadow-[inset_0_1px_0_rgba(255,255,255,0.7),0_16px_36px_rgba(0,0,0,0.18)]">
              <div className="flex flex-col gap-3 border-b border-slate-200 bg-slate-50/95 px-4 py-3 text-slate-700 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#00b896]" />
                  <span>{currentNote ? `현재 목표음 ${currentNote}` : "피치 그래프"}</span>
                </div>

                <div className="flex items-center gap-4 text-xs font-medium text-slate-500">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-0.5 w-5 rounded-full bg-[#2563eb]" />
                    내 피치
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-0.5 w-5 rounded-full border-t-2 border-dashed border-[#ef476f]" />
                    목표음
                  </span>
                </div>
              </div>

              <canvas
                ref={canvasRef}
                width={1200}
                height={520}
                className="block w-full h-[310px] md:h-[440px] xl:h-[490px] bg-white"
              />
            </div>

            <div className="pt-6 flex justify-center">
              <button
                onClick={() => (isRunning ? stopAll() : startSequence())}
                disabled={isBusy}
                className={`group w-[84px] h-[84px] rounded-full border border-white/20 bg-[radial-gradient(circle_at_30%_30%,#2de3bf,#00b896)] text-white shadow-[0_16px_35px_rgba(0,0,0,0.35)] flex items-center justify-center transition
                ${
                  isBusy
                    ? "opacity-60 cursor-not-allowed"
                    : "hover:scale-[1.03] active:scale-95"
                }`}
                aria-label={isRunning ? "테스트 중지" : "테스트 시작"}
              >
                {isRunning ? (
                  <Square className="w-8 h-8" />
                ) : (
                  <Mic className="w-8 h-8 group-hover:rotate-3 transition-transform" />
                )}
              </button>
            </div>
          </div>
        </section>

        {tessitura && (
          <section className="rounded-2xl border border-emerald-300/25 bg-emerald-400/10 backdrop-blur-xl p-5 md:p-6">
            <div className="text-sm text-emerald-100/90">분석된 테시투라</div>
            <div className="mt-2 text-2xl font-bold text-white">
              {tessitura.low} - {tessitura.high}
            </div>
            <div className="mt-1 text-sm text-emerald-100/80">
              평균 강도 {(tessitura.avgStrong * 100).toFixed(1)}%
            </div>
          </section>
        )}

        <section className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden">
          <div className="px-5 md:px-6 py-4 border-b border-white/10 flex items-center justify-between">
            <div className="font-semibold text-white">측정 결과</div>
            <div className="text-xs text-white/60">Strong / Weak 기준 판정</div>
          </div>

          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-black/35 backdrop-blur-md">
                <tr className="text-white/75">
                  <th className="text-left px-5 md:px-6 py-3 font-medium">음</th>
                  <th className="text-left px-5 md:px-6 py-3 font-medium">Strong%</th>
                  <th className="text-left px-5 md:px-6 py-3 font-medium">Weak%</th>
                  <th className="text-left px-5 md:px-6 py-3 font-medium">판정</th>
                </tr>
              </thead>

              <tbody>
                {results.map((r, i) => {
                  const rowBg =
                    r.grade === "Strong OK"
                      ? "bg-emerald-500/12"
                      : r.grade === "Weak OK"
                      ? "bg-amber-500/12"
                      : "bg-rose-500/12";

                  return (
                    <tr key={i} className={`border-t border-white/10 ${rowBg}`}>
                      <td className="px-5 md:px-6 py-3.5 text-white font-medium">{r.note}</td>
                      <td className="px-5 md:px-6 py-3.5 text-white/90">{(r.strong * 100).toFixed(0)}%</td>
                      <td className="px-5 md:px-6 py-3.5 text-white/90">{(r.weak * 100).toFixed(0)}%</td>
                      <td className="px-5 md:px-6 py-3.5">
                        <span
                          className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium border ${
                            r.grade === "Strong OK"
                              ? "border-emerald-300/35 text-emerald-100 bg-emerald-500/25"
                              : r.grade === "Weak OK"
                              ? "border-amber-300/35 text-amber-100 bg-amber-500/25"
                              : "border-rose-300/35 text-rose-100 bg-rose-500/25"
                          }`}
                        >
                          {r.grade}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            {results.length === 0 && (
              <div className="px-6 py-12 text-center text-sm text-white/60">
                아직 결과가 없습니다. 마이크 버튼을 눌러 테스트를 시작해 주세요.
              </div>
            )}
          </div>
        </section>

        {status === "done" && (
          <section className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-5 md:p-6">
            <div className="text-base md:text-lg font-semibold text-white">재도전 가능한 음 (음당 1회)</div>
            <p className="mt-2 text-sm text-white/70 leading-relaxed">
              Strong OK 경계에 인접한 Weak OK, 그리고 Strong 구간 내부의 Weak OK/Fail 음만 재측정할 수 있습니다.
            </p>

            <div className="mt-4">
              {(() => {
                const strongIndices = results
                  .map((r, i) => ({ i, grade: r.grade }))
                  .filter((x) => x.grade === "Strong OK")
                  .map((x) => x.i);

                if (strongIndices.length < 1) {
                  return <p className="text-sm text-white/65">Strong OK 음이 없어 재도전할 수 없습니다.</p>;
                }

                const minStrong = Math.min(...strongIndices);
                const maxStrong = Math.max(...strongIndices);

                const internal = results
                  .map((r, i) => ({ ...r, i }))
                  .filter(
                    (x) =>
                      x.i > minStrong &&
                      x.i < maxStrong &&
                      (x.grade === "Weak OK" || x.grade === "Fail")
                  );

                const lower = [];
                for (let i = minStrong - 1; i >= 0; i--) {
                  const r = results[i];
                  if (!r || r.grade !== "Weak OK") break;
                  lower.push({ ...r, i });
                }

                const higher = [];
                for (let i = maxStrong + 1; i < results.length; i++) {
                  const r = results[i];
                  if (!r || r.grade !== "Weak OK") break;
                  higher.push({ ...r, i });
                }

                const candidates = [...lower.reverse(), ...internal, ...higher];

                if (candidates.length === 0) {
                  return <p className="text-sm text-white/65">재도전 가능한 음이 없습니다.</p>;
                }

                return (
                  <div className="flex flex-wrap gap-2.5">
                    {candidates.map((c) => (
                      <button
                        key={c.note}
                        onClick={() => retryNote(c.note)}
                        disabled={retryingNote !== null || retriedNotes.includes(c.note)}
                        className={`px-3.5 py-2 rounded-xl border border-white/15 bg-white/10 hover:bg-white/15 transition text-sm ${
                          retryingNote !== null || retriedNotes.includes(c.note)
                            ? "opacity-60 cursor-not-allowed"
                            : ""
                        }`}
                      >
                        {retryingNote === c.note
                          ? `${c.note} 재측정 중...`
                          : retriedNotes.includes(c.note)
                          ? `${c.note} 재도전 완료`
                          : `${c.note} 재도전`}
                      </button>
                    ))}
                  </div>
                );
              })()}
            </div>
          </section>
        )}

        {downloadUrl && (
          <section className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-5 md:p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="font-semibold text-white">녹음 파일</div>
              <div className="text-sm text-white/70">테스트 음성을 WAV로 다운로드할 수 있습니다.</div>
            </div>

            <a
              href={downloadUrl}
              download={`pitchtest_${Date.now()}.wav`}
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 transition text-white"
            >
              <Download className="w-4 h-4" />
              WAV 다운로드
            </a>
          </section>
        )}
      </div>
    </div>
  );
}
