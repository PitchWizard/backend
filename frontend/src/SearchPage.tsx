import { useEffect, useState } from "react";
import { AlertCircle, ArrowLeft, Loader2, Music2, Search } from "lucide-react";
import { getSongCatalog } from "./api/songApi";

type Props = {
  onBack: () => void;
  isDarkMode: boolean;
  onSelectSong?: (song: SongItem) => void;
  onSelectSinger?: (name: string) => void;
};

type SongItem = {
  id: string | number;
  title: string;
  artist: string;
  album?: string;
  duration?: string;
  key?: string;
  coverUrl?: string;
  midiMin?: number | null;
  midiMedian?: number | null;
  midiMax?: number | null;
  rmsMean?: number | null;
  rmsStd?: number | null;
};

const MOCK_SONGS: SongItem[] = [
  { id: "mock-1", title: "Love wins all", artist: "아이유", album: "The Winning", duration: "4:31", midiMin: 54, midiMedian: 65, midiMax: 79 },
  { id: "mock-2", title: "밤편지", artist: "아이유", album: "Palette", duration: "4:14", midiMin: 52, midiMedian: 62, midiMax: 74 },
  { id: "mock-3", title: "사건의 지평선", artist: "윤하", album: "YOUNHA 6th", duration: "5:00", midiMin: 55, midiMedian: 66, midiMax: 80 },
  { id: "mock-4", title: "주저하는 연인들을 위해", artist: "잔나비", album: "전설", duration: "4:25", midiMin: 49, midiMedian: 60, midiMax: 71 },
  { id: "mock-5", title: "Super Shy", artist: "NewJeans", album: "Get Up", duration: "2:35", midiMin: 57, midiMedian: 64, midiMax: 74 },
  { id: "mock-6", title: "Hype Boy", artist: "NewJeans", album: "New Jeans", duration: "2:59", midiMin: 56, midiMedian: 64, midiMax: 75 },
  { id: "mock-7", title: "너의 모든 순간", artist: "성시경", album: "별에서 온 그대 OST", duration: "4:05", midiMin: 45, midiMedian: 55, midiMax: 67 },
  { id: "mock-8", title: "좋니", artist: "윤종신", album: "LISTEN 010", duration: "6:10", midiMin: 46, midiMedian: 56, midiMax: 68 },
  { id: "mock-9", title: "밤양갱", artist: "비비", album: "밤양갱", duration: "2:26", midiMin: 53, midiMedian: 61, midiMax: 72 },
];

function midiToNoteName(midi: number) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const rounded = Math.round(midi);
  return `${names[((rounded % 12) + 12) % 12]}${Math.floor(rounded / 12) - 1}`;
}

function formatRangeLabel(song: SongItem) {
  const { midiMin: min, midiMax: max } = song;
  if (typeof min === "number" && typeof max === "number") return `${midiToNoteName(min)} ~ ${midiToNoteName(max)}`;
  return song.key ? `키 ${song.key}` : "";
}

export default function SearchPage({ onBack, isDarkMode, onSelectSong, onSelectSinger }: Props) {
  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<SongItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    setLoading(true);
    getSongCatalog()
      .then((list) => { setCatalog(list); setError(""); setNotice(""); })
      .catch(() => { setCatalog(MOCK_SONGS); setNotice("백엔드 연결이 없어 샘플 데이터로 표시 중입니다."); })
      .finally(() => setLoading(false));
  }, []);

  const dark = isDarkMode;
  const bg = dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]";
  const text = dark ? "text-white" : "text-[#1d1d1f]";
  const sub = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const card = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";
  const cardHover = dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]";
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";

  const filtered = catalog.filter((song) => {
    const q = query.toLowerCase();
    return !q || song.title.toLowerCase().includes(q) || song.artist.toLowerCase().includes(q);
  });

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

        <main className="flex-1 flex flex-col pt-16 px-8 pb-16 max-w-6xl mx-auto w-full">
          {/* 히어로 */}
          <div className="text-center pt-16 pb-10">
            <div className={`inline-flex items-center gap-2 text-[12px] px-3 py-1 rounded-full border ${border} ${card} ${sub} mb-7`}>
              <span className="w-1.5 h-1.5 rounded-full bg-[#00d9b1] animate-pulse" />
              Song Library
            </div>
            <h2 className={`text-[52px] font-bold leading-[1.05] tracking-tight ${text} mb-4`}>
              내 목소리에 맞는<br />
              <span className="text-[#00d9b1]">노래를 찾아보세요</span>
            </h2>
            <p className={`text-[16px] ${sub} leading-relaxed max-w-md mx-auto`}>
              음역대 데이터를 기반으로 어울리는 곡을 추천합니다
            </p>
          </div>

          {/* 검색 바 */}
          <div className={`flex items-center gap-3 rounded-2xl border ${border} ${card} px-5 py-3.5 mb-5`}>
            <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${dark ? "bg-white/[0.06]" : "bg-black/[0.05]"}`}>
              <Search className={`w-4 h-4 ${sub}`} />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="노래 제목이나 가수를 검색해보세요"
              className={`w-full bg-transparent outline-none text-[15px] ${text} ${dark ? "placeholder:text-white/25" : "placeholder:text-black/25"}`}
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-[13px] text-red-400 mb-4">
              <AlertCircle className="h-4 w-4 flex-shrink-0" /><span>{error}</span>
            </div>
          )}
          {notice && (
            <div className={`flex items-center gap-2 rounded-xl border px-4 py-3 text-[13px] mb-4 ${dark ? "border-amber-300/20 bg-amber-400/[0.08] text-amber-300" : "border-amber-500/25 bg-amber-100/50 text-amber-700"}`}>
              <AlertCircle className="h-4 w-4 flex-shrink-0" /><span>{notice}</span>
            </div>
          )}

          <section>
            {loading ? (
              <div className={`rounded-2xl border ${border} ${card} px-5 py-14 text-center`}>
                <Loader2 className={`mx-auto h-5 w-5 animate-spin ${sub} mb-3`} />
                <p className={`text-[13px] ${sub}`}>곡 목록을 불러오는 중입니다...</p>
              </div>
            ) : filtered.length === 0 ? (
              <div className={`rounded-2xl border ${border} ${card} px-5 py-14 text-center`}>
                <p className={`text-[13px] ${sub}`}>검색 결과가 없습니다.</p>
              </div>
            ) : (
              <div className="space-y-2">
                <p className={`text-[12px] ${sub} mb-3`}>총 {filtered.length}곡</p>
                {filtered.map((song) => (
                  <button
                    type="button"
                    onClick={() => onSelectSong?.(song)}
                    key={song.id}
                    className={`w-full text-left rounded-2xl border ${border} ${card} p-5 transition-all ${cardHover} hover:border-[#00d9b1]/20 group`}
                  >
                    <div className="flex items-center gap-4 justify-between">
                      <div className="flex items-center gap-4 min-w-0">
                        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border ${border} ${dark ? "bg-white/[0.05]" : "bg-black/[0.04]"} group-hover:border-[#00d9b1]/30 transition-colors`}>
                          <Music2 className={`h-4 w-4 ${sub}`} />
                        </div>
                        <div className="min-w-0">
                          <p className={`truncate text-[16px] font-semibold ${text}`}>{song.title}</p>
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); onSelectSinger?.(song.artist); }}
                            className={`text-[13px] ${sub} hover:text-[#00d9b1] transition-colors`}
                          >
                            {song.artist}
                          </button>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1.5 justify-end flex-shrink-0">
                        {formatRangeLabel(song) && (
                          <span className="rounded-full border border-[#00d9b1]/30 px-2.5 py-0.5 text-[11px] text-[#00d9b1]">{formatRangeLabel(song)}</span>
                        )}
                        {typeof song.midiMedian === "number" && (
                          <span className={`rounded-full border ${border} px-2.5 py-0.5 text-[11px] ${sub}`}>중앙 {midiToNoteName(song.midiMedian)}</span>
                        )}
                        {song.duration && (
                          <span className={`rounded-full border ${border} px-2.5 py-0.5 text-[11px] ${sub}`}>{song.duration}</span>
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
