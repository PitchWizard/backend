import { useEffect, useState } from "react";
import { ArrowLeft, BarChart3, Music2 } from "lucide-react";
import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

function midiToNote(midi: number): string {
  const names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
  return names[midi % 12] + (Math.floor(midi / 12) - 1);
}

const WHITE_KEY_NOTES = ["C","D","E","F","G","A","B"];
const WHITE_KEY_SEMITONES = [0, 2, 4, 5, 7, 9, 11];

type WhiteKey = { note: string; midi: number; hasBlackRight: boolean };
const whiteKeys: WhiteKey[] = [];
for (let oct = 3; oct <= 5; oct++) {
  WHITE_KEY_NOTES.forEach((n, i) => {
    whiteKeys.push({ note: n + oct, midi: (oct + 1) * 12 + WHITE_KEY_SEMITONES[i], hasBlackRight: n !== "E" && n !== "B" });
  });
}
whiteKeys.push({ note: "C6", midi: 84, hasBlackRight: false });

type Song = { song_id: number; title: string; artist: string; midi_min: number; midi_median: number; midi_max: number };
type Props = { singerName: string; onBack: () => void; isDarkMode: boolean };

export default function SingerPage({ singerName, onBack, isDarkMode }: Props) {
  const [songs, setSongs] = useState<Song[]>([]);

  useEffect(() => {
    axios.get(`${BASE_URL}/songs`).then((r) => {
      setSongs(r.data.filter((s: Song) => s.artist === singerName));
    }).catch(() => {});
  }, [singerName]);

  const validSongs = songs.filter((s) => s.midi_min && s.midi_max);
  const avgMin = validSongs.length ? Math.round(validSongs.reduce((a, s) => a + s.midi_min, 0) / validSongs.length) : null;
  const avgMax = validSongs.length ? Math.round(validSongs.reduce((a, s) => a + s.midi_max, 0) / validSongs.length) : null;
  const avgMedian = validSongs.length ? Math.round(validSongs.reduce((a, s) => a + s.midi_median, 0) / validSongs.length) : null;
  const lowNote = avgMin != null ? midiToNote(avgMin) : null;
  const highNote = avgMax != null ? midiToNote(avgMax) : null;
  const hasMeasured = avgMin != null && avgMax != null;

  const dark = isDarkMode;
  const bg = dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]";
  const text = dark ? "text-white" : "text-[#1d1d1f]";
  const sub = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const card = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";
  const cardHover = dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]";
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";
  const innerCard = dark ? "bg-white/[0.03]" : "bg-black/[0.02]";

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
              Singer Voice Range
            </div>
            <h2 className={`text-[52px] font-bold leading-[1.05] tracking-tight ${text} mb-4`}>
              {singerName}
            </h2>
            <p className={`text-[16px] ${sub} leading-relaxed`}>
              평균 음역대 분석과 수록곡 목록
            </p>
          </div>

          <div className="space-y-4">
            <section className={`rounded-2xl border ${border} ${card} p-7`}>
              <div className="flex items-center gap-3 mb-6">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${border} ${innerCard}`}>
                  <BarChart3 className={`w-4 h-4 ${sub}`} />
                </div>
                <h3 className={`text-[20px] font-semibold ${text}`}>평균 음역대</h3>
              </div>

              <div className={`rounded-xl border ${border} ${innerCard} p-6 mb-4`}>
                <p className={`text-[12px] tracking-wide ${sub} mb-2`}>음역대 범위</p>
                <p className={`text-[48px] font-bold leading-none ${text}`}>
                  {hasMeasured ? `${lowNote} ~ ${highNote}` : "데이터 없음"}
                </p>
                <div className={`mt-5 rounded-xl border ${border} ${dark ? "bg-black/30" : "bg-white/80"} p-4`}>
                  <div className="relative h-[110px] rounded-xl border border-black/15 overflow-hidden bg-gradient-to-b from-white to-[#f0f0f0]">
                    <div className="absolute inset-0 flex">
                      {whiteKeys.map((key, index) => {
                        const whiteInRange = hasMeasured && key.midi >= avgMin! && key.midi <= avgMax!;
                        const blackInRange = hasMeasured && (key.midi + 1) >= avgMin! && (key.midi + 1) <= avgMax!;
                        return (
                          <div key={key.note} className={`relative flex-1 border-r last:border-r-0 ${whiteInRange ? "bg-gradient-to-b from-[#b8ffef] to-[#83f5d8] border-black/20" : "bg-gradient-to-b from-white to-[#ececec] border-black/15"}`}>
                            {key.hasBlackRight && index < whiteKeys.length - 1 && (
                              <span className={`absolute right-0 top-0 translate-x-1/2 z-10 h-[66px] w-[54%] rounded-b-md border border-black/50 shadow-[0_7px_10px_rgba(0,0,0,0.35)] ${blackInRange ? "bg-gradient-to-b from-[#00f3c8] to-[#00b894]" : "bg-gradient-to-b from-[#262626] to-black"}`} />
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  <div className={`mt-3 flex justify-between text-[11px] ${sub}`}>
                    <span>C3</span>
                    <span className="text-[#00d9b1] font-semibold">{hasMeasured ? `${lowNote} ~ ${highNote}` : "데이터 없음"}</span>
                    <span>C6</span>
                  </div>
                </div>
              </div>

              {avgMedian && (
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { label: "최저음 (평균)", value: lowNote },
                    { label: "중앙음 (평균)", value: midiToNote(avgMedian) },
                    { label: "최고음 (평균)", value: highNote },
                  ].map((item) => (
                    <div key={item.label} className={`rounded-xl border ${border} ${innerCard} p-4 text-center`}>
                      <p className={`text-[11px] ${sub} mb-1.5`}>{item.label}</p>
                      <p className="text-[24px] font-bold text-[#00d9b1]">{item.value ?? "--"}</p>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className={`rounded-2xl border ${border} ${card} p-7`}>
              <div className="flex items-center gap-3 mb-5">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${border} ${innerCard}`}>
                  <Music2 className={`w-4 h-4 ${sub}`} />
                </div>
                <h3 className={`text-[20px] font-semibold ${text}`}>수록곡 ({songs.length})</h3>
              </div>
              {songs.length === 0 ? (
                <p className={`text-[13px] ${sub}`}>등록된 곡이 없습니다.</p>
              ) : (
                <div className="space-y-2">
                  {songs.map((song) => (
                    <div key={song.song_id} className={`flex items-center justify-between gap-4 rounded-xl border ${border} ${innerCard} px-5 py-3.5`}>
                      <div className="flex items-center gap-3 min-w-0">
                        <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${border} ${dark ? "bg-white/[0.05]" : "bg-black/[0.04]"}`}>
                          <Music2 className={`h-3.5 w-3.5 ${sub}`} />
                        </div>
                        <p className={`truncate text-[15px] font-medium ${text}`}>{song.title}</p>
                      </div>
                      <div className="flex shrink-0 gap-1.5">
                        {song.midi_min != null && song.midi_max != null && (
                          <span className="rounded-full border border-[#00d9b1]/30 px-2.5 py-0.5 text-[11px] text-[#00d9b1]">
                            {midiToNote(Math.round(song.midi_min))} ~ {midiToNote(Math.round(song.midi_max))}
                          </span>
                        )}
                        {song.midi_median != null && (
                          <span className={`rounded-full border ${border} px-2.5 py-0.5 text-[11px] ${sub}`}>중앙 {midiToNote(Math.round(song.midi_median))}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
