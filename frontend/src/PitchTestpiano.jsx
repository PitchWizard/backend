// PitchTestpiano.jsx
import React, { useEffect, useRef, useState } from "react";
import saveVocalRange from "./api/vocalRangeApi";   // ✅ default import


const DEFAULTS = {
  measureWindowSec: 2.5, // 사용자의 음정 측정 시간
  voiceOnsetRmsThreshold: 0.015,
  frameIntervalMs: 60,  // frame 간격
  strongCents: 40,  // strong 판정 기준 (기준음과의 차이가 40 cents 이내)
  weakCents: 75,  // weak 판정 기준 (기준음과의 차이가 75 cents 이내)
  strongPercent: 0.6, // strong 판정 기준 (전체 프레임 중 strong 프레임의 비율 == 60% 이상)
  weakPercent: 0.4,  // weak 판정 기준 (전체 프레임 중 weak 프레임의 비율 == 40% 이상)
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

export default function PitchTestPiano() {
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
  const [tessitura, setTessitura] = useState(null); // ✅ 추가: 테시투라 상태
  // ✅ 음별 재도전 관리 (각 음별 1회)
  const [retriedNotes, setRetriedNotes] = useState([]); // 이미 재시도한 음 리스트
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
      const { freq, rms } = autocorrelate(timeDomain, ctx.sampleRate);
      console.log("RMS:", rms, "Freq:", freq);

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
    // 새 테스트 시작할 때 이전 재시도 기록 초기화
    setRetriedNotes([]);
    setRetryingNote(null);
    await initAudio();
    startRecording();

    const res = [];
    for (const n of NOTES_TO_TEST) {
      const r = await runNoteTest(n);
      res.push(r);
      setResults([...res]);
      await new Promise((s) => setTimeout(s, 400));
    }

    stopRecording();

    // 테시투라 분석
    const { tessitura, segments } = estimateTessitura(res, {
      strongThreshold: DEFAULTS.strongPercent,
      minNotes: 3,
      maxAllowedGaps: 1,
    });
    setTessitura(tessitura);
    console.log("🎼 Tessitura 분석 결과:", tessitura);
    console.log("📊 모든 구간:", segments);

        // MIDI 값 계산
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

    // 🔹 avg_rms는 아직 따로 계산 안 하니까, 일단 null로 보내고 나중에 원하면 계산 로직 추가
    const avgRms = null;

    // 🔹 서버 전송용 payload (백엔드 /vocal-range 스키마에 맞춤)
    const payload = tessitura
      ? {
          user_id: 1,              // TODO: 로그인 붙이면 실제 유저 id로 교체
          midi_min,
          midi_median,
          midi_max,
          low_note: tessitura.low,
          high_note: tessitura.high,
          avg_rms: avgRms,
        }
      : null;

    console.log(
      "📤 서버로 전송할 테시투라 데이터:",
      JSON.stringify(payload, null, 2)
    );

    // 🔹 백엔드로 전송
    if (payload) {
      try {
        const res = await saveVocalRange(payload);
        console.log("✅ 음역 정보 저장 완료:", res);
      } catch (err) {
        console.error("❌ 음역 정보 저장 실패:", err);
      }
    }

    setStatus("done");

  }


  // 재도전 함수
  async function retryNote(noteName) {
    if (retriedNotes.includes(noteName)) {
      alert(`${noteName} 음은 이미 재도전했습니다.`);
      return;
    }

    const noteObj = NOTES_TO_TEST.find(n => n.note === noteName);
    if (!noteObj) return;

    const cur = results.find(r => r.note === noteName);
    if (!cur || (cur.grade !== "Weak OK" && cur.grade !== "Fail")) {
      alert("재도전은 Weak OK 또는 Fail인 음만 가능합니다.");
      return;
    }

    // 성공한 범위 계산
    const successGrades = ["Strong OK", "Weak OK"];
    const successIndices = results
      .map((r, i) => ({ i, grade: r.grade }))
      .filter(x => successGrades.includes(x.grade))
      .map(x => x.i);

    if (successIndices.length === 0) {
      alert("성공한 음이 없어서 재도전 대상 범위를 계산할 수 없습니다.");
      return;
    }

    const noteIndex = NOTES_TO_TEST.findIndex(n => n.note === noteName);
    const minSucc = Math.min(...successIndices);
    const maxSucc = Math.max(...successIndices);


    // ===== 재도전 실행 =====
    setRetryingNote(noteName);
    setStatus("retrying");

    try {
      await initAudio();     // 🎵 오디오 초기화 (이제 playTone 가능)
      startRecording();

      const updated = await runNoteTest(noteObj); // 제시음 재생 + 사용자 입력 측정

      stopRecording();
      stopAll(); // 오디오 종료

      // 결과 갱신
      setResults(prev => {
        const next = prev.map(r => (r.note === noteName ? updated : r));
        const { tessitura: newTessitura } = estimateTessitura(next, {
          strongThreshold: DEFAULTS.strongPercent,
          minNotes: 3,
          maxAllowedGaps: 1,
        });
        setTessitura(newTessitura);
        return next;
      });

      // ✅ 이 음은 재시도 완료 목록에 추가
      setRetriedNotes(prev => [...prev, noteName]);
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

  // 그래프 C2~C7
  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width,
      height = canvas.height;
    ctx.clearRect(0, 0, width, height);
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
        ctx.fillText(midiToNoteName(m), 5, y - 2);
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

    ctx.strokeStyle = "blue";
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

  return (
    <div
      style={{
        fontFamily: "sans-serif",
        maxWidth: 1000,
        margin: "0 auto",
        padding: 16,
        textAlign: "center"
      }}
    >

      <h2>사용자 음역대 테스트</h2>
      <div>
        <button onClick={startSequence} disabled={status === "running"}>
          테스트 시작
        </button>
        <button onClick={stopAll}>중단</button>
      </div>
      <div>
        상태: {status} {currentNote && `(현재: ${currentNote})`}
      </div>

    <div style={{ display: "flex", alignItems: "flex-start", gap: 20, marginTop: 10 }}>
      {/* 왼쪽: 그래프 (크기 그대로 유지) */}
      <canvas
        ref={canvasRef}
        width={900}
        height={600}
        style={{ border: "1px solid black" }}
      />

      {/* 오른쪽: 표 컨테이너 */}
      <div
        style={{
          flexShrink: 0,
          maxHeight: 600,            // 그래프 높이에 맞춤
          overflowY: "auto",         // 표가 길면 스크롤
          border: "1px solid #ccc",  // 표 테두리 구분용 (선택)
          padding: 8,                // 표 주변 여백
          background: "white",       // 캔버스 배경과 구분
          minWidth: 300              // 폭 최소 확보 (모양 깨짐 방지)
        }}
      >
        <h3 style={{ marginTop: 0 }}>결과 테이블</h3>
        <table
          border="1"
          cellPadding="5"
          style={{
            borderCollapse: "collapse",
            width: "100%",            // 표 폭이 꽉 차도록
            textAlign: "center"
          }}
        >
          <thead style={{ background: "#f0f0f0" }}>
            <tr>
              <th>음</th>
              <th>Strong%</th>
              <th>Weak%</th>
              <th>판정</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr
                key={i}
                style={{
                  background:
                    r.grade === "Strong OK"
                      ? "#9cff9c"
                      : r.grade === "Weak OK"
                      ? "#ffe699"
                      : "#ff9999",
                }}
              >
                <td>{r.note}</td>
                <td>{(r.strong * 100).toFixed(0)}%</td>
                <td>{(r.weak * 100).toFixed(0)}%</td>
                <td>{r.grade}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* 🎯 재도전 UI: 테스트 완료 상태일 때 */}
        {status === "done" && (
          <div style={{ marginTop: 16 }}>
            <h3>🎯 재도전 가능한 음 (음별 1회)</h3>
            <p style={{ marginTop: 6, marginBottom: 6 }}>
              <strong>최저음/최고음 경계에 인접한 Weak OK</strong> 음과,<br></br>  
              <strong>Strong OK 범위 내부의 Weak OK / Fail</strong> 음만 재도전할 수 있습니다.
            </p>


            <div>
              {(() => {
              // 1️⃣ Strong OK 인덱스 찾기
              const strongIndices = results
                .map((r, i) => ({ i, grade: r.grade }))
                .filter(x => x.grade === "Strong OK")
                .map(x => x.i);

              if (strongIndices.length < 1)
                return <p>Strong OK 음이 없어 재도전할 수 없습니다.</p>;

              const minStrong = Math.min(...strongIndices);
              const maxStrong = Math.max(...strongIndices);

              // 2️⃣ 내부 약/실패 음 (Strong OK 사이)
              const internal = results
                .map((r, i) => ({ ...r, i }))
                .filter(
                  x =>
                    x.i > minStrong &&
                    x.i < maxStrong &&
                    (x.grade === "Weak OK" || x.grade === "Fail")
                );

              // 3️⃣ 하단 인접 Weak OK 연속 구간 (Strong OK 최저음보다 낮은)
              const lower = [];
              for (let i = minStrong - 1; i >= 0; i--) {
                const r = results[i];
                if (!r || r.grade !== "Weak OK") break;
                lower.push({ ...r, i });
              }

              // 4️⃣ 상단 인접 Weak OK 연속 구간 (Strong OK 최고음보다 낮은)
              const higher = [];
              for (let i = maxStrong + 1; i < results.length; i++) {
                const r = results[i];
                if (!r || r.grade !== "Weak OK") break;
                higher.push({ ...r, i });
              }

              // 5️⃣ 전체 후보 합치기
              const candidates = [...lower.reverse(), ...internal, ...higher];

              if (candidates.length === 0)
                return <p>재도전 가능한 음이 없습니다.</p>;

              return (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {candidates.map(c => (
                    <button
                      key={c.note}
                      onClick={() => retryNote(c.note)}
                      disabled={retryingNote !== null || retriedNotes.includes(c.note)}
                      style={{ padding: "6px 10px" }}
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

      </div>
    </div>

      {tessitura && (
        <div style={{ marginTop: 20 }}>
          <h3>🎤 분석된 테시투라</h3>
          <p>
            <strong>
              {tessitura.low} ~ {tessitura.high}
            </strong>{" "}
            (평균 강도 {(tessitura.avgStrong * 100).toFixed(1)}%)
          </p>
        </div>
      )}

      {downloadUrl && (
        <div style={{ marginTop: 12 }}>
          <h4>녹음 파일</h4>
          <a href={downloadUrl} download={`pitchtest_${Date.now()}.wav`}>
            WAV 다운로드
          </a>
        </div>
      )}
    </div>
  );
}
