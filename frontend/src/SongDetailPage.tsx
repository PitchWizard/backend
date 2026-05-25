import { ArrowLeft, Music2, PlayCircle, UserRound } from "lucide-react";

type SongInfo = {
  id: string | number;
  title: string;
  artist: string;
  album?: string;
  duration?: string;
  midiMin?: number | null;
  midiMedian?: number | null;
  midiMax?: number | null;
  rmsMean?: number | null;
  rmsStd?: number | null;
};

type Props = {
  onBack: () => void;
  onGoAccompaniment: () => void;
  isDarkMode: boolean;
  user: any;
  song: SongInfo;
};

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function midiToNoteName(midi: number) {
  const rounded = Math.round(midi);
  return `${NOTE_NAMES[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`;
}

function buildWhiteMidiKeys() {
  const keys: number[] = [];
  for (let midi = 36; midi <= 84; midi++) if (!NOTE_NAMES[midi % 12].includes("#")) keys.push(midi);
  return keys;
}

function hasBlackKeyToRight(whiteMidi: number) {
  const note = NOTE_NAMES[whiteMidi % 12];
  return note !== "E" && note !== "B";
}

function clampToAccompanimentRange(v: number) { return Math.max(-5, Math.min(5, v)); }
function inRange(midi: number, min: number | null, max: number | null) {
  return min !== null && max !== null && midi >= min && midi <= max;
}

function keyClass(inSong: boolean, inUser: boolean, isBlack: boolean) {
  if (inSong && inUser) return isBlack ? "bg-gradient-to-b from-[#00f4c9] to-[#00c89f]" : "bg-gradient-to-b from-[#b9ffef] to-[#82f1d6]";
  if (inSong) return isBlack ? "bg-gradient-to-b from-[#ffd37a] to-[#ffad33]" : "bg-gradient-to-b from-[#ffe8bc] to-[#ffd388]";
  if (inUser) return isBlack ? "bg-gradient-to-b from-[#79d2ff] to-[#3b9fff]" : "bg-gradient-to-b from-[#c5edff] to-[#8ad4ff]";
  return isBlack ? "bg-gradient-to-b from-[#2a2a2a] to-black" : "bg-gradient-to-b from-white to-[#ececec]";
}

export default function SongDetailPage({ onBack, onGoAccompaniment, isDarkMode, user, song }: Props) {
  const dark = isDarkMode;
  const bg = dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]";
  const text = dark ? "text-white" : "text-[#1d1d1f]";
  const sub = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const card = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";
  const cardHover = dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]";
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";
  const innerCard = dark ? "bg-white/[0.03]" : "bg-black/[0.02]";

  const songMin = typeof song.midiMin === "number" ? song.midiMin : null;
  const songMedian = typeof song.midiMedian === "number" ? song.midiMedian : null;
  const songMax = typeof song.midiMax === "number" ? song.midiMax : null;
  const userMin = typeof user?.midi_min === "number" && user.midi_min > 0 ? user.midi_min : null;
  const userMedian = typeof user?.midi_median === "number" && user.midi_median > 0 ? user.midi_median : null;
  const userMax = typeof user?.midi_max === "number" && user.midi_max > 0 ? user.midi_max : null;
  const hasSongRange = songMin !== null && songMax !== null;
  const hasUserRange = userMin !== null && userMax !== null;
  const overlapMin = hasSongRange && hasUserRange ? Math.max(songMin, userMin) : null;
  const overlapMax = hasSongRange && hasUserRange ? Math.min(songMax, userMax) : null;
  const hasOverlap = overlapMin !== null && overlapMax !== null && overlapMin <= overlapMax;
  const recommendedShift = typeof songMedian === "number" && typeof userMedian === "number"
    ? clampToAccompanimentRange(Math.round(userMedian - songMedian)) : null;
  const whiteKeys = buildWhiteMidiKeys();

  return (
    <div className={`min-h-screen ${bg} relative overflow-hidden`}>
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] rounded-full blur-[120px] opacity-20 bg-[#00d9b1]" />
        <div className={`absolute bottom-0 right-0 w-[400px] h-[400px] rounded-full blur-[100px] opacity-10 ${dark ? "bg-blue-500" : "bg-blue-400"}`} />
      </div>

      <div className="relative z-10 min-h-screen flex flex-col font-['Pretendard']">
        <header className={`fixed top-0 left-0 right-0 z-50 ${dark ? "bg-[#0a0a0a]/80" : "bg-[#f5f5f7]/80"} backdrop-blur-xl border-b ${border}`}>
          <div className="max-w-6xl mx-auto px-8 h-16 flex items-center justify-between">
            <button onClick={onBack} className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${card} ${cardHover} border ${border}`}>
              <ArrowLeft className={`w-4 h-4 ${sub}`} />
            </button>
            <span className={`text-[15px] font-semibold tracking-tight ${text}`}>PitchWizard</span>
            <div className="w-8" />
          </div>
        </header>

        <main className="flex-1 pt-16 px-8 pb-16 max-w-6xl mx-auto w-full">
          {/* 히어로 */}
          <div className="text-center pt-16 pb-10">
            <div className={`inline-flex items-center gap-2 text-[12px] px-3 py-1 rounded-full border ${border} ${card} ${sub} mb-7`}>
              <span className="w-1.5 h-1.5 rounded-full bg-[#00d9b1] animate-pulse" />
              Song Profile
            </div>
            <h2 className={`text-[48px] font-bold leading-[1.05] tracking-tight ${text} mb-3`}>
              {song.title}
            </h2>
            <p className={`text-[18px] ${sub}`}>{song.artist}</p>
            <div className="flex flex-wrap items-center justify-center gap-2 mt-5">
              {hasSongRange && (
                <span className="rounded-full border border-[#00d9b1]/30 px-3 py-1 text-[12px] text-[#00d9b1]">
                  음역 {midiToNoteName(songMin)} ~ {midiToNoteName(songMax)}
                </span>
              )}
              {typeof songMedian === "number" && (
                <span className={`rounded-full border ${border} px-3 py-1 text-[12px] ${sub}`}>중앙 {midiToNoteName(songMedian)}</span>
              )}
              {song.album && <span className={`rounded-full border ${border} px-3 py-1 text-[12px] ${sub}`}>앨범 {song.album}</span>}
              {song.duration && <span className={`rounded-full border ${border} px-3 py-1 text-[12px] ${sub}`}>{song.duration}</span>}
            </div>
          </div>

          <div className="space-y-4">
            {/* 음역대 비교 */}
            <section className={`rounded-2xl border ${border} ${card} p-7`}>
              <div className="flex items-center gap-3 mb-5">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${border} ${innerCard}`}>
                  <Music2 className={`w-4 h-4 ${sub}`} />
                </div>
                <h3 className={`text-[20px] font-semibold ${text}`}>음역대 비교</h3>
              </div>

              <div className="flex flex-wrap gap-2 mb-5">
                <span className="rounded-full border border-[#ffbf59]/30 px-2.5 py-0.5 text-[11px] text-[#ffcf7d]">곡 음역대</span>
                <span className="rounded-full border border-[#63c8ff]/30 px-2.5 py-0.5 text-[11px] text-[#9edfff]">내 음역대</span>
                <span className="rounded-full border border-[#00d9b1]/30 px-2.5 py-0.5 text-[11px] text-[#00d9b1]">겹치는 구간</span>
              </div>

              <div className={`rounded-xl border ${border} ${innerCard} p-4 mb-5`}>
                <div className="relative h-[120px] rounded-xl border border-black/15 overflow-hidden bg-gradient-to-b from-white to-[#f0f0f0]">
                  <div className="absolute inset-0 flex">
                    {whiteKeys.map((whiteMidi, index) => {
                      const whiteInSong = inRange(whiteMidi, songMin, songMax);
                      const whiteInUser = inRange(whiteMidi, userMin, userMax);
                      const blackMidi = whiteMidi + 1;
                      const blackInSong = hasBlackKeyToRight(whiteMidi) && inRange(blackMidi, songMin, songMax);
                      const blackInUser = hasBlackKeyToRight(whiteMidi) && inRange(blackMidi, userMin, userMax);
                      return (
                        <div key={`white-${whiteMidi}`} className={`relative flex-1 border-r last:border-r-0 border-black/15 ${keyClass(whiteInSong, whiteInUser, false)}`}>
                          {hasBlackKeyToRight(whiteMidi) && index < whiteKeys.length - 1 && (
                            <span className={`absolute right-0 top-0 translate-x-1/2 z-10 h-[72px] w-[54%] rounded-b-md border border-black/50 shadow-[0_7px_10px_rgba(0,0,0,0.35)] ${keyClass(blackInSong, blackInUser, true)}`} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className={`mt-3 flex justify-between text-[11px] ${sub}`}><span>C2</span><span>C4</span><span>C6</span></div>
              </div>

              <div className="grid gap-3 md:grid-cols-2 mb-4">
                <div className={`rounded-xl border ${border} ${innerCard} p-4`}>
                  <p className={`text-[12px] ${sub}`}>곡 음역대</p>
                  <p className={`mt-1 text-[20px] font-semibold ${text}`}>{hasSongRange ? `${midiToNoteName(songMin)} ~ ${midiToNoteName(songMax)}` : "정보 없음"}</p>
                </div>
                <div className={`rounded-xl border ${border} ${innerCard} p-4`}>
                  <p className={`text-[12px] ${sub}`}>내 음역대</p>
                  <p className={`mt-1 text-[20px] font-semibold ${text}`}>{hasUserRange ? `${midiToNoteName(userMin)} ~ ${midiToNoteName(userMax)}` : "아직 측정 전"}</p>
                </div>
              </div>

              <div className={`rounded-xl border ${border} ${innerCard} p-5`}>
                <div className="flex items-center gap-2 mb-3">
                  <UserRound className={`h-4 w-4 ${sub}`} />
                  <p className={`text-[16px] font-semibold ${text}`}>비교 해석</p>
                </div>
                {!hasUserRange ? (
                  <p className={`text-[14px] leading-7 ${sub}`}>내 음역대 데이터가 아직 없습니다. 먼저 음역대 테스트를 완료해주세요.</p>
                ) : (
                  <div className="space-y-1.5">
                    <p className={`text-[14px] leading-7 ${sub}`}>
                      {hasOverlap ? `겹치는 구간은 ${midiToNoteName(overlapMin)} ~ ${midiToNoteName(overlapMax)}입니다.` : "현재 측정값 기준으로는 겹치는 구간이 거의 없습니다."}
                    </p>
                    <p className={`text-[14px] leading-7 ${sub}`}>
                      추천 전조는{" "}
                      <span className="font-semibold text-[#00d9b1]">
                        {recommendedShift === null ? "계산 불가" : recommendedShift > 0 ? `+${recommendedShift}키` : `${recommendedShift}키`}
                      </span>입니다.
                    </p>
                  </div>
                )}
              </div>
            </section>

            {/* 반주 이동 */}
            <section className={`rounded-2xl border ${border} ${card} p-6`}>
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className={`text-[17px] font-semibold ${text}`}>반주 재생으로 이어가기</p>
                  <p className={`mt-1 text-[13px] ${sub}`}>
                    선택한 곡을 반주 페이지에서 바로 불러옵니다
                    {recommendedShift !== null ? ` · 추천 ${recommendedShift > 0 ? `+${recommendedShift}` : recommendedShift}키` : ""}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={onGoAccompaniment}
                  className="inline-flex items-center gap-2 px-6 py-2.5 rounded-full bg-[#00d9b1] text-white text-[14px] font-semibold shadow-lg shadow-[#00d9b1]/20 hover:shadow-[#00d9b1]/40 hover:scale-[1.02] active:scale-[0.99] transition-all duration-200 flex-shrink-0"
                >
                  <PlayCircle className="h-4 w-4" />
                  반주 페이지로 이동
                </button>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
