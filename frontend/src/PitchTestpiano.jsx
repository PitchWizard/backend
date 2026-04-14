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

    ctx.clearRect(0, 0, width, height);

    // 흰 배경 유지 (어두운 UI 위에서도 가독성 확보)
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const minMidi = 36; // C2
    const maxMidi = 96; // C7

    for (let m = minMidi; m <= maxMidi; m++) {
      const y = ((maxMidi - m) / (maxMidi - minMidi)) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.strokeStyle = m % 12 === 0 ? "#777" : "#ddd";
      ctx.lineWidth = m % 12 === 0 ? 2 : 1;
      ctx.stroke();
      if (m % 12 === 0) {
        ctx.fillStyle = "black";
        ctx.font = "12px sans-serif";
        ctx.fillText(midiToNoteName(m), 8, y - 4);
      }
    }

    if (currentNote) {
      const noteObj = NOTES_TO_TEST.find((x) => x.note === currentNote);
      if (noteObj) {
        const y = ((maxMidi - noteObj.midi) / (maxMidi - minMidi)) * height;
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    ctx.strokeStyle = "#0b5cff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    pitchHistory.forEach((f, i) => {
      const m = freqToMidi(f);
      const y = ((maxMidi - m) / (maxMidi - minMidi)) * height;
      const x = (i / Math.max(1, pitchHistory.length - 1)) * width;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const isRunning = status === "running";
  const isBusy = status === "retrying";

  return (
    <div className="relative">
      {/* 상단 큰 캔버스 영역 (스크린샷처럼 ‘큰 유리 카드’ 안에 꽉) */}
      <div className="p-8 md:p-10">
        <div className="rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden">
          <div className="p-6 md:p-8">
            <canvas
              ref={canvasRef}
              width={1200}
              height={520}
              className="w-full h-[340px] md:h-[420px] rounded-2xl bg-white"
            />

            {/* 상태 텍스트 (작게) */}
            <div className="mt-5 text-center text-sm text-white/70">
              상태: <span className="text-white/90">{status}</span>{" "}
              {currentNote && (
                <span className="text-white/60"> · 현재: {currentNote}</span>
              )}
            </div>
          </div>

          {/* 하단 마이크 버튼 (스크린샷 스타일) */}
          <div className="pb-10 flex justify-center">
            <button
              onClick={() => (isRunning ? stopAll() : startSequence())}
              disabled={isBusy}
              className={`w-20 h-20 rounded-full border border-white/15 bg-white/10 backdrop-blur-xl shadow-2xl shadow-black/40
                flex items-center justify-center transition
                ${isBusy ? "opacity-60 cursor-not-allowed" : "hover:bg-white/15 active:scale-95"}
              `}
              aria-label={isRunning ? "중단" : "테스트 시작"}
            >
              {isRunning ? (
                <Square className="w-7 h-7 text-white" />
              ) : (
                <Mic className="w-7 h-7 text-white" />
              )}
            </button>
          </div>
        </div>

        {/* 기능은 그대로 유지: 결과/재도전/테시투라/다운로드를 ‘아래’에 표시 */}
        <div className="mt-8 space-y-6">
          {/* 테시투라 */}
          {tessitura && (
            <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-6 text-white">
              <div className="text-lg font-semibold">🎤 분석된 테시투라</div>
              <div className="mt-2 text-white/80">
                <span className="font-semibold text-white">
                  {tessitura.low} ~ {tessitura.high}
                </span>{" "}
                <span className="text-white/60">
                  (평균 강도 {(tessitura.avgStrong * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
          )}

          {/* 결과 테이블 */}
          <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl overflow-hidden">
            <div className="px-6 py-4 flex items-center justify-between">
              <div className="text-white font-semibold">결과</div>
              <div className="text-xs text-white/60">
                Strong / Weak 판정 표
              </div>
            </div>

            <div className="max-h-[420px] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-black/30 backdrop-blur-md">
                  <tr className="text-white/80">
                    <th className="text-left px-6 py-3 font-medium">음</th>
                    <th className="text-left px-6 py-3 font-medium">Strong%</th>
                    <th className="text-left px-6 py-3 font-medium">Weak%</th>
                    <th className="text-left px-6 py-3 font-medium">판정</th>
                  </tr>
                </thead>

                <tbody>
                  {results.map((r, i) => {
                    const rowBg =
                      r.grade === "Strong OK"
                        ? "bg-emerald-500/20"
                        : r.grade === "Weak OK"
                        ? "bg-amber-500/20"
                        : "bg-rose-500/20";

                    return (
                      <tr key={i} className={`border-t border-white/10 ${rowBg}`}>
                        <td className="px-6 py-3 text-white font-medium">{r.note}</td>
                        <td className="px-6 py-3 text-white/90">
                          {(r.strong * 100).toFixed(0)}%
                        </td>
                        <td className="px-6 py-3 text-white/90">
                          {(r.weak * 100).toFixed(0)}%
                        </td>
                        <td className="px-6 py-3 text-white/90">{r.grade}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {results.length === 0 && (
                <div className="px-6 py-10 text-center text-white/60">
                  아직 결과가 없습니다. 마이크 버튼을 눌러 테스트를 시작하세요.
                </div>
              )}
            </div>
          </div>

          {/* 재도전 UI (기능 그대로) */}
          {status === "done" && (
            <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-6 text-white">
              <div className="text-lg font-semibold">🎯 재도전 가능한 음 (음별 1회)</div>
              <p className="mt-2 text-sm text-white/70 leading-relaxed">
                <strong className="text-white">최저음/최고음 경계에 인접한 Weak OK</strong> 음과,
                <br />
                <strong className="text-white">Strong OK 범위 내부의 Weak OK / Fail</strong> 음만 재도전할 수 있습니다.
              </p>

              <div className="mt-4">
                {(() => {
                  const strongIndices = results
                    .map((r, i) => ({ i, grade: r.grade }))
                    .filter((x) => x.grade === "Strong OK")
                    .map((x) => x.i);

                  if (strongIndices.length < 1)
                    return <p className="text-white/70">Strong OK 음이 없어 재도전할 수 없습니다.</p>;

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

                  if (candidates.length === 0)
                    return <p className="text-white/70">재도전 가능한 음이 없습니다.</p>;

                  return (
                    <div className="flex flex-wrap gap-2">
                      {candidates.map((c) => (
                        <button
                          key={c.note}
                          onClick={() => retryNote(c.note)}
                          disabled={retryingNote !== null || retriedNotes.includes(c.note)}
                          className={`px-3 py-2 rounded-xl border border-white/15 bg-white/10 hover:bg-white/15 transition text-sm
                            ${
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
            </div>
          )}

          {/* 다운로드 (기능 그대로) */}
          {downloadUrl && (
            <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-6 text-white flex items-center justify-between gap-4">
              <div>
                <div className="font-semibold">녹음 파일</div>
                <div className="text-sm text-white/70">WAV로 다운로드할 수 있어요.</div>
              </div>

              <a
                href={downloadUrl}
                download={`pitchtest_${Date.now()}.wav`}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/15 transition"
              >
                <Download className="w-4 h-4" />
                WAV 다운로드
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}