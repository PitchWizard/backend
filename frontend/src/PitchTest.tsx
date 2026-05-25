import PitchTestpiano from "./PitchTestpiano";
import { ArrowLeft, LogIn } from "lucide-react";

type Props = {
  onBack: () => void;
  isDarkMode: boolean;
  user: any;
  onTestComplete: (updated: object) => void;
};

export default function PitchTest({ onBack, isDarkMode, user, onTestComplete }: Props) {
  const dark = isDarkMode;
  const bg = dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]";
  const text = dark ? "text-white" : "text-[#1d1d1f]";
  const sub = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const card = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";
  const cardHover = dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]";
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";

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

        <main className="flex-1 pt-16 px-8 pb-12 max-w-6xl mx-auto w-full">
          {/* 히어로 */}
          <div className="text-center pt-16 pb-10">
            <div className={`inline-flex items-center gap-2 text-[12px] px-3 py-1 rounded-full border ${border} ${card} ${sub} mb-7`}>
              <span className="w-1.5 h-1.5 rounded-full bg-[#00d9b1] animate-pulse" />
              Vocal Range Check
            </div>
            <h2 className={`text-[52px] font-bold leading-[1.05] tracking-tight ${text} mb-4`}>
              나의 음역대를<br />
              <span className="text-[#00d9b1]">테스트해보세요</span>
            </h2>
            <p className={`text-[16px] ${sub} leading-relaxed max-w-md mx-auto`}>
              마이크 버튼을 누른 뒤 재생되는 기준음을 따라 불러주세요
            </p>
          </div>

          {!user ? (
            <div className={`flex flex-col items-center justify-center gap-5 py-20 rounded-2xl border ${border} ${card}`}>
              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center border ${border} ${dark ? "bg-white/5" : "bg-black/5"}`}>
                <LogIn className={`w-7 h-7 ${sub}`} />
              </div>
              <div className="text-center">
                <p className={`text-[20px] font-semibold ${text}`}>로그인이 필요한 기능입니다</p>
                <p className={`text-[14px] ${sub} mt-1.5`}>측정 결과를 저장하려면 먼저 로그인해 주세요.</p>
              </div>
              <button
                onClick={onBack}
                className="px-7 py-3 rounded-full bg-[#00d9b1] text-white font-semibold text-[14px] shadow-lg shadow-[#00d9b1]/20 hover:shadow-[#00d9b1]/40 hover:scale-[1.02] active:scale-[0.99] transition-all duration-200"
              >
                홈으로 돌아가기
              </button>
            </div>
          ) : (
            <div className={`rounded-2xl border ${border} ${card} overflow-hidden`}>
              <PitchTestpiano userId={user.id} onTestComplete={onTestComplete} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
