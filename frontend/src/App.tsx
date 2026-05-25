import { useState } from "react";
import {
  CheckCircle,
  Headphones,
  LogIn,
  Moon,
  Music2,
  Search,
  Smile,
  Sun,
  VolumeX,
  Waves,
  X,
} from "lucide-react";
import AccompanimentPage from "./AccompanimentPage.tsx";
import LoginPage from "./LoginPage.tsx";
import PitchTest from "./PitchTest.tsx";
import SearchPage from "./SearchPage.tsx";
import SingerPage from "./SingerPage.tsx";
import SongDetailPage from "./SongDetailPage.tsx";
import SignupPage from "./SignupPage.tsx";
import VoiceRangePage from "./VoiceRangePage.tsx";

type Page = "home" | "login" | "signup" | "test" | "search" | "songDetail" | "accompaniment" | "range" | "singer";

type SelectedSong = {
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

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [selectedSong, setSelectedSong] = useState<SelectedSong | null>(null);
  const [selectedSinger, setSelectedSinger] = useState<string | null>(null);
  const [singerFrom, setSingerFrom] = useState<Page>("home");
  const [user, setUser] = useState<any>(() => {
    const saved = localStorage.getItem("user");
    return saved ? JSON.parse(saved) : null;
  });

  function handleLogin(userData: any) { setUser(userData); }
  function handleLogout() { localStorage.removeItem("user"); setUser(null); }

  if (currentPage === "login") return <LoginPage onBack={() => setCurrentPage("home")} onLogin={handleLogin} onGoSignup={() => setCurrentPage("signup")} isDarkMode={isDarkMode} />;
  if (currentPage === "signup") return <SignupPage onBack={() => setCurrentPage("home")} onGoLogin={() => setCurrentPage("login")} isDarkMode={isDarkMode} />;
  if (currentPage === "test") return <PitchTest onBack={() => setCurrentPage("home")} isDarkMode={isDarkMode} user={user} onTestComplete={(updated: any) => { const u = { ...user, ...updated }; setUser(u); localStorage.setItem("user", JSON.stringify(u)); }} />;
  if (currentPage === "search") return <SearchPage onBack={() => setCurrentPage("home")} isDarkMode={isDarkMode} onSelectSong={(song) => { setSelectedSong(song); setCurrentPage("songDetail"); }} onSelectSinger={(name) => { setSelectedSinger(name); setSingerFrom("search"); setCurrentPage("singer"); }} />;
  if (currentPage === "songDetail" && selectedSong) return <SongDetailPage onBack={() => setCurrentPage("search")} onGoAccompaniment={() => setCurrentPage("accompaniment")} isDarkMode={isDarkMode} user={user} song={selectedSong} />;
  if (currentPage === "songDetail" && !selectedSong) return <SearchPage onBack={() => setCurrentPage("home")} isDarkMode={isDarkMode} onSelectSong={(song) => { setSelectedSong(song); setCurrentPage("songDetail"); }} />;
  if (currentPage === "accompaniment") return <AccompanimentPage onBack={() => setCurrentPage("home")} isDarkMode={isDarkMode} user={user} initialSongId={selectedSong?.id} initialSongTitle={selectedSong?.title} initialSongArtist={selectedSong?.artist} />;
  if (currentPage === "range") return <VoiceRangePage onBack={() => setCurrentPage("home")} isDarkMode={isDarkMode} user={user} onSelectSinger={(name) => { setSelectedSinger(name); setSingerFrom("range"); setCurrentPage("singer"); }} />;
  if (currentPage === "singer" && selectedSinger) return <SingerPage singerName={selectedSinger} onBack={() => setCurrentPage(singerFrom)} isDarkMode={isDarkMode} />;

  const dark = isDarkMode;
  const bg = dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]";
  const text = dark ? "text-white" : "text-[#1d1d1f]";
  const sub = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const card = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";
  const cardHover = dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]";
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";

  const features = [
    {
      icon: <Music2 className="w-5 h-5" />,
      title: "나의 음역대",
      desc: "테스트 결과와 유사 가수를 확인합니다",
      page: "range" as Page,
    },
    {
      icon: <Search className="w-5 h-5" />,
      title: "노래 검색",
      desc: "내 목소리에 맞는 노래를 찾습니다",
      page: "search" as Page,
    },
    {
      icon: <Waves className="w-5 h-5" />,
      title: "실시간 피치",
      desc: "반주에 맞춰 내 피치를 분석합니다",
      page: "accompaniment" as Page,
    },
  ];

  return (
    <div className={`min-h-screen ${bg} relative overflow-hidden`}>
      {/* 배경 그라디언트 */}
      <div className="absolute inset-0 pointer-events-none">
        <div className={`absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] rounded-full blur-[120px] opacity-20 ${dark ? "bg-[#00d9b1]" : "bg-[#00d9b1]"}`} />
        <div className={`absolute bottom-0 right-0 w-[400px] h-[400px] rounded-full blur-[100px] opacity-10 ${dark ? "bg-blue-500" : "bg-blue-400"}`} />
      </div>

      <div className="relative z-10 flex flex-col min-h-screen">
        {/* 헤더 */}
        <header className={`fixed top-0 left-0 right-0 z-50 ${dark ? "bg-[#0a0a0a]/80" : "bg-[#f5f5f7]/80"} backdrop-blur-xl border-b ${border}`}>
          <div className="max-w-6xl mx-auto px-8 h-16 flex items-center justify-between">

            {/* 로고 영역 */}
            <div className="flex items-center gap-2.5">
              <div className={`w-8 h-8 rounded-lg overflow-hidden border ${border} flex items-center justify-center flex-shrink-0 ${dark ? "bg-white/5" : "bg-black/5"}`}>
                {/* ↓ /public/logo.png 파일 추가 시 자동 표시 */}
                <img
                  src="/logo.png"
                  alt="logo"
                  className="w-full h-full object-contain"
                  onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                />
              </div>
              <span className={`text-[15px] font-semibold tracking-tight ${text}`}>PitchWizard</span>
            </div>

            {/* 우측 컨트롤 */}
            <div className="flex items-center gap-2">
              {user ? (
                <>
                  <span className={`text-[13px] ${sub} mr-1`}>{user.username}</span>
                  <button
                    onClick={handleLogout}
                    className={`text-[13px] px-3.5 py-1.5 rounded-full border ${border} ${card} ${text} transition-all ${cardHover}`}
                  >
                    로그아웃
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setCurrentPage("login")}
                  className={`text-[13px] px-3.5 py-1.5 rounded-full border ${border} ${card} ${text} transition-all ${cardHover} flex items-center gap-1.5`}
                >
                  <LogIn className="w-3.5 h-3.5" />
                  로그인
                </button>
              )}
              <button
                onClick={() => setIsDarkMode(!isDarkMode)}
                className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${card} ${cardHover} border ${border}`}
              >
                {dark ? <Sun className={`w-4 h-4 ${sub}`} /> : <Moon className={`w-4 h-4 ${sub}`} />}
              </button>
            </div>
          </div>
        </header>

        {/* 히어로 */}
        <main className="flex-1 flex flex-col items-center justify-center pt-16 px-8">
          <div className="max-w-3xl w-full mx-auto text-center pt-28 pb-20">

            {/* 뱃지 */}
            <div className={`inline-flex items-center gap-2 text-[12px] px-3 py-1 rounded-full border ${border} ${card} ${sub} mb-8`}>
              <span className="w-1.5 h-1.5 rounded-full bg-[#00d9b1] animate-pulse" />
              AI 기반 음역대 분석
            </div>

            <h1 className={`font-['Pretendard'] text-[64px] md:text-[80px] font-bold leading-[1.05] tracking-tight ${text} mb-5`}>
              자신의 목소리를<br />
              <span className="text-[#00d9b1]">정확하게</span> 파악하세요
            </h1>

            <p className={`text-[17px] ${sub} mb-10 leading-relaxed max-w-xl mx-auto`}>
              음역대 테스트부터 실시간 피치 분석까지,<br />더 잘 노래할 수 있도록 도와드립니다
            </p>

            <div className="flex items-center justify-center gap-3 flex-wrap">
              <button
                onClick={() => setShowModal(true)}
                className="px-8 py-3.5 rounded-full bg-[#00d9b1] text-white text-[15px] font-semibold shadow-lg shadow-[#00d9b1]/20 hover:shadow-[#00d9b1]/40 hover:scale-[1.02] active:scale-[0.99] transition-all duration-200"
              >
                음역대 테스트 시작
              </button>
              <button
                onClick={() => setCurrentPage("search")}
                className={`px-8 py-3.5 rounded-full border ${border} ${card} ${text} text-[15px] font-medium transition-all ${cardHover} hover:scale-[1.02]`}
              >
                노래 검색
              </button>
            </div>

            <div className={`flex items-center justify-center gap-6 mt-10 text-[13px] ${sub}`}>
              {["5분 소요", "무료", "간편한 분석"].map((label) => (
                <span key={label} className="flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5 text-[#00d9b1]" />
                  {label}
                </span>
              ))}
            </div>
          </div>

          {/* 기능 카드 */}
          <div className="max-w-5xl w-full mx-auto grid md:grid-cols-3 gap-4 pb-24 px-4">
            {features.map((f) => (
              <button
                key={f.page}
                onClick={() => setCurrentPage(f.page)}
                className={`text-left p-6 rounded-2xl border ${border} ${card} ${cardHover} transition-all group hover:border-[#00d9b1]/30`}
              >
                <div className={`w-9 h-9 rounded-xl flex items-center justify-center mb-4 border ${border} ${dark ? "bg-white/5 text-white/70" : "bg-black/5 text-black/60"} group-hover:text-[#00d9b1] group-hover:border-[#00d9b1]/30 transition-colors`}>
                  {f.icon}
                </div>
                <h3 className={`text-[16px] font-semibold mb-1.5 ${text}`}>{f.title}</h3>
                <p className={`text-[13px] leading-relaxed ${sub}`}>{f.desc}</p>
              </button>
            ))}
          </div>
        </main>
      </div>

      {/* 테스트 안내 모달 */}
      {showModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setShowModal(false)} />
          <div className={`relative w-full max-w-lg ${dark ? "bg-[#111]" : "bg-white"} rounded-3xl border ${border} shadow-2xl overflow-hidden`}>

            <div className="p-8 pb-6">
              <button onClick={() => setShowModal(false)} className={`absolute top-5 right-5 w-8 h-8 rounded-full flex items-center justify-center ${card} ${cardHover} transition-all`}>
                <X className={`w-4 h-4 ${sub}`} />
              </button>
              <h2 className={`text-[24px] font-bold ${text} mb-1`}>테스트 시작 전 안내</h2>
              <p className={`text-[14px] ${sub}`}>정확한 측정을 위해 아래 사항을 확인해주세요</p>
            </div>

            <div className={`mx-6 mb-6 space-y-2 p-4 rounded-2xl border ${border} ${card}`}>
              {[
                { icon: <VolumeX className="w-4 h-4" />, title: "조용한 환경", desc: "주변 소음이 적은 공간을 선택하세요" },
                { icon: <Smile className="w-4 h-4" />, title: "편안한 발성", desc: "무리하지 않고 편안하게 불러주세요" },
                { icon: <Headphones className="w-4 h-4" />, title: "헤드폰 권장", desc: "이어폰 착용 시 더 정확한 측정이 가능합니다" },
              ].map((item) => (
                <div key={item.title} className={`flex items-start gap-3 p-3 rounded-xl ${cardHover} transition-all`}>
                  <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${dark ? "bg-white/8 text-white/60" : "bg-black/5 text-black/50"}`}>
                    {item.icon}
                  </div>
                  <div>
                    <p className={`text-[13px] font-semibold ${text}`}>{item.title}</p>
                    <p className={`text-[12px] ${sub} mt-0.5`}>{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="px-6 pb-6">
              <button
                onClick={() => { setShowModal(false); setCurrentPage("test"); }}
                className="w-full py-4 rounded-2xl bg-[#00d9b1] text-white text-[16px] font-bold shadow-lg shadow-[#00d9b1]/20 hover:shadow-[#00d9b1]/40 hover:scale-[1.01] active:scale-[0.99] transition-all duration-200"
              >
                테스트 시작
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
