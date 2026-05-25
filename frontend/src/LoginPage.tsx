import { useState } from "react";
import { ArrowLeft, Lock, UserRound } from "lucide-react";
import { login } from "./api/authApi";

type Props = {
  onBack: () => void;
  onLogin?: (user: object) => void;
  onGoSignup?: () => void;
  isDarkMode: boolean;
};

export default function LoginPage({ onBack, onLogin, onGoSignup, isDarkMode }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const dark = isDarkMode;
  const bg = dark ? "bg-[#0a0a0a]" : "bg-[#f5f5f7]";
  const text = dark ? "text-white" : "text-[#1d1d1f]";
  const sub = dark ? "text-white/50" : "text-[#1d1d1f]/50";
  const card = dark ? "bg-white/[0.04]" : "bg-black/[0.03]";
  const cardHover = dark ? "hover:bg-white/[0.07]" : "hover:bg-black/[0.06]";
  const border = dark ? "border-white/[0.08]" : "border-black/[0.08]";
  const inputBg = dark ? "bg-white/[0.05]" : "bg-black/[0.04]";
  const ph = dark ? "placeholder:text-white/25" : "placeholder:text-black/25";

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const user = await login(username, password);
      if (rememberMe) localStorage.setItem("user", JSON.stringify(user));
      onLogin?.(user);
      onBack();
    } catch (err: any) {
      setError(err.response?.data?.detail || "로그인에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  }

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

        <main className="flex-1 flex items-center justify-center pt-16 px-6 py-12">
          <div className="w-full max-w-[440px]">
            {/* 히어로 */}
            <div className="text-center mb-10">
              <div className={`inline-flex items-center gap-2 text-[12px] px-3 py-1 rounded-full border ${border} ${card} ${sub} mb-6`}>
                <span className="w-1.5 h-1.5 rounded-full bg-[#00d9b1] animate-pulse" />
                Account Access
              </div>
              <h2 className={`text-[40px] font-bold leading-[1.05] tracking-tight ${text} mb-2`}>
                다시 만나서<br />
                <span className="text-[#00d9b1]">반가워요</span>
              </h2>
              <p className={`text-[15px] ${sub}`}>계속하려면 로그인해주세요</p>
            </div>

            <form className="space-y-4" onSubmit={handleSubmit}>
              <label className="block">
                <span className={`mb-1.5 block text-[13px] font-medium ${sub}`}>아이디</span>
                <div className={`flex items-center gap-3 rounded-xl border ${border} ${inputBg} px-4 py-3.5`}>
                  <UserRound className={`w-4 h-4 ${sub} flex-shrink-0`} />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="아이디를 입력하세요"
                    className={`w-full bg-transparent outline-none text-[15px] ${text} ${ph}`}
                  />
                </div>
              </label>

              <label className="block">
                <span className={`mb-1.5 block text-[13px] font-medium ${sub}`}>비밀번호</span>
                <div className={`flex items-center gap-3 rounded-xl border ${border} ${inputBg} px-4 py-3.5`}>
                  <Lock className={`w-4 h-4 ${sub} flex-shrink-0`} />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="비밀번호를 입력하세요"
                    className={`w-full bg-transparent outline-none text-[15px] ${text} ${ph}`}
                  />
                </div>
              </label>

              <div className="flex items-center justify-between pt-0.5">
                <label className={`flex items-center gap-2 text-[13px] ${sub} cursor-pointer`}>
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={() => setRememberMe((v) => !v)}
                    className="h-3.5 w-3.5 rounded accent-[#00d9b1]"
                  />
                  로그인 상태 유지
                </label>
                <button type="button" className="text-[13px] font-medium text-[#00d9b1]">비밀번호 찾기</button>
              </div>

              {error && <p className="text-[13px] text-red-400 text-center">{error}</p>}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3.5 rounded-xl bg-[#00d9b1] text-white text-[15px] font-semibold shadow-lg shadow-[#00d9b1]/20 hover:shadow-[#00d9b1]/40 hover:scale-[1.01] active:scale-[0.99] transition-all duration-200 disabled:opacity-60"
              >
                {loading ? "로그인 중..." : "로그인하기"}
              </button>
            </form>

            <div className="my-6 flex items-center gap-4">
              <div className={`h-px flex-1 border-t ${border}`} />
              <span className={`text-[11px] uppercase tracking-[0.3em] ${sub}`}>or</span>
              <div className={`h-px flex-1 border-t ${border}`} />
            </div>

            <div className="grid gap-2.5">
              <button type="button" className={`w-full rounded-xl border ${border} ${card} ${cardHover} py-3.5 text-[14px] font-medium ${text} transition-all`}>
                Google로 계속하기
              </button>
              <button type="button" className={`w-full rounded-xl border ${border} ${card} ${cardHover} py-3.5 text-[14px] font-medium ${text} transition-all`}>
                카카오로 계속하기
              </button>
            </div>

            <p className={`mt-6 text-center text-[13px] ${sub}`}>
              계정이 없으신가요?{" "}
              <button type="button" onClick={onGoSignup} className="font-semibold text-[#00d9b1]">회원가입</button>
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
