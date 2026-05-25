import { useState } from "react";
import { ArrowLeft, Check, Lock, Mail, UserRound, X } from "lucide-react";
import { signup } from "./api/authApi";

type Props = {
  onBack: () => void;
  onGoLogin?: () => void;
  isDarkMode: boolean;
};

export default function SignupPage({ onBack, onGoLogin, isDarkMode }: Props) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
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

  function isValidPassword(v: string) {
    return v.length >= 8 && /[A-Z]/.test(v) && /[a-z]/.test(v) && /\d/.test(v);
  }
  const passwordChecks = [
    { label: "8자 이상", valid: password.length >= 8 },
    { label: "영문 대문자", valid: /[A-Z]/.test(password) },
    { label: "영문 소문자", valid: /[a-z]/.test(password) },
    { label: "숫자 포함", valid: /\d/.test(password) },
  ];

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(""); setSuccess("");
    const normalizedEmail = email.trim();
    if (!username || !normalizedEmail || !password || !confirmPassword) { setError("모든 항목을 입력해 주세요."); return; }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(normalizedEmail)) { setError("올바른 이메일 형식을 입력해 주세요."); return; }
    if (!isValidPassword(password)) { setError("비밀번호 규칙을 확인해 주세요."); return; }
    if (password !== confirmPassword) { setError("비밀번호 확인이 일치하지 않습니다."); return; }
    setLoading(true);
    try {
      await signup({ username, email: normalizedEmail, password });
      setSuccess("회원가입이 완료되었습니다. 로그인해 주세요.");
      setTimeout(() => onGoLogin?.(), 900);
    } catch (err: any) {
      setError(err.response?.data?.detail || "회원가입에 실패했습니다.");
    } finally { setLoading(false); }
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
                Create Account
              </div>
              <h2 className={`text-[40px] font-bold leading-[1.05] tracking-tight ${text} mb-2`}>
                시작해볼까요,<br />
                <span className="text-[#00d9b1]">회원가입</span>
              </h2>
              <p className={`text-[15px] ${sub}`}>새 계정을 만들어 음역대를 분석해보세요</p>
            </div>

            <form className="space-y-4" onSubmit={handleSubmit}>
              <label className="block">
                <span className={`mb-1.5 block text-[13px] font-medium ${sub}`}>아이디</span>
                <div className={`flex items-center gap-3 rounded-xl border ${border} ${inputBg} px-4 py-3.5`}>
                  <UserRound className={`w-4 h-4 ${sub} flex-shrink-0`} />
                  <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="아이디를 입력하세요" className={`w-full bg-transparent outline-none text-[15px] ${text} ${ph}`} />
                </div>
              </label>

              <label className="block">
                <span className={`mb-1.5 block text-[13px] font-medium ${sub}`}>이메일</span>
                <div className={`flex items-center gap-3 rounded-xl border ${border} ${inputBg} px-4 py-3.5`}>
                  <Mail className={`w-4 h-4 ${sub} flex-shrink-0`} />
                  <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" className={`w-full bg-transparent outline-none text-[15px] ${text} ${ph}`} />
                </div>
              </label>

              <label className="block">
                <span className={`mb-1.5 block text-[13px] font-medium ${sub}`}>비밀번호</span>
                <div className={`flex items-center gap-3 rounded-xl border ${border} ${inputBg} px-4 py-3.5`}>
                  <Lock className={`w-4 h-4 ${sub} flex-shrink-0`} />
                  <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="비밀번호를 입력하세요" className={`w-full bg-transparent outline-none text-[15px] ${text} ${ph}`} />
                </div>
                {password.length > 0 && (
                  <div className="mt-2.5 grid grid-cols-2 gap-1.5">
                    {passwordChecks.map((item) => (
                      <div key={item.label} className="flex items-center gap-1.5 text-[12px]">
                        {item.valid ? <Check className="h-3.5 w-3.5 text-[#00d9b1]" /> : <X className={`h-3.5 w-3.5 ${sub}`} />}
                        <span className={item.valid ? "text-[#00d9b1]" : sub}>{item.label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </label>

              <label className="block">
                <span className={`mb-1.5 block text-[13px] font-medium ${sub}`}>비밀번호 확인</span>
                <div className={`flex items-center gap-3 rounded-xl border ${border} ${inputBg} px-4 py-3.5`}>
                  <Lock className={`w-4 h-4 ${sub} flex-shrink-0`} />
                  <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="비밀번호를 다시 입력하세요" className={`w-full bg-transparent outline-none text-[15px] ${text} ${ph}`} />
                </div>
              </label>

              {error && <p className="text-[13px] text-red-400 text-center">{error}</p>}
              {success && <p className="text-[13px] text-[#00d9b1] text-center">{success}</p>}

              <button type="submit" disabled={loading} className="w-full py-3.5 rounded-xl bg-[#00d9b1] text-white text-[15px] font-semibold shadow-lg shadow-[#00d9b1]/20 hover:shadow-[#00d9b1]/40 hover:scale-[1.01] active:scale-[0.99] transition-all duration-200 disabled:opacity-60">
                {loading ? "가입 중..." : "회원가입"}
              </button>
            </form>

            <div className="my-6 flex items-center gap-4">
              <div className={`h-px flex-1 border-t ${border}`} />
              <span className={`text-[11px] uppercase tracking-[0.3em] ${sub}`}>or</span>
              <div className={`h-px flex-1 border-t ${border}`} />
            </div>

            <div className="grid gap-2.5 opacity-40 cursor-not-allowed">
              <button type="button" disabled className={`w-full rounded-xl border ${border} ${card} py-3.5 text-[14px] font-medium ${text}`}>Google로 계속하기 (준비 중)</button>
              <button type="button" disabled className={`w-full rounded-xl border ${border} ${card} py-3.5 text-[14px] font-medium ${text}`}>카카오로 계속하기 (준비 중)</button>
            </div>

            <p className={`mt-6 text-center text-[13px] ${sub}`}>
              이미 계정이 있으신가요?{" "}
              <button type="button" onClick={onGoLogin} className="font-semibold text-[#00d9b1]">로그인</button>
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
