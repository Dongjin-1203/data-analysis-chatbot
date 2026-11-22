"""에이전트용 터미널 명령 실행.

- 세션별 작업 폴더에서 실행하고 시간 제한을 둔다.
- PATH 앞에 챗봇 가상환경을 두어 `python` 이 코드 실행 도구와 같은 환경을 가리키게 한다.
- 가상환경에는 pip 가 없으므로 `pip install ...` / `python -m pip install ...` 은
  `uv pip install ...` 로 바꿔 실행한다(VIRTUAL_ENV 로 대상 환경 지정).
- 명백히 위험한 명령은 막지만, 이것은 최소한의 안전장치일 뿐 보안 샌드박스가 아니다.
"""

import locale
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 8000

SHELL_NAME = "cmd.exe (Windows)" if os.name == "nt" else "sh (POSIX)"

# 파일·시스템을 망가뜨리거나 외부로 영향을 주는 명령 (대소문자 무시)
BLOCKED_PATTERNS = [
    r"\brm\s+(-\w*\s+)*-\w*[rf]",  # rm -rf, rm -r
    r"\brmdir\b",
    r"\brd\s+/s",
    r"\bdel\b",
    r"\berase\b",
    r"\bremove-item\b",
    r"\bformat\b\s+[a-z]:",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\breg\s+(delete|add)\b",
    r"\bnet\s+user\b",
    r"\btakeown\b",
    r"\bicacls\b",
    r"\bchmod\s+-R\b",
    r"\bchown\b",
    r"\bsudo\b",
    r"\bgit\s+(push|reset|clean)\b",
    r"(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|iex|python)",
    r"\b(pip|uv\s+pip)\s+uninstall\b",
    r"\buv\s+(sync|remove)\b",
    r":\(\)\s*\{",  # fork bomb
]

PIP_INSTALL = re.compile(r"^\s*(?:python(?:3)?\s+-m\s+)?pip3?\s+install\b", re.IGNORECASE)


@dataclass
class ShellResult:
    ok: bool
    exit_code: int | None
    output: str
    command: str
    blocked: bool = False

    def to_text(self) -> str:
        if self.blocked:
            return f"[실행 거부] {self.output}"
        head = "[명령 성공]" if self.ok else f"[명령 실패] 종료 코드 {self.exit_code}"
        body = self.output.rstrip() or "(출력 없음)"
        note = f"\n실제 실행한 명령: {self.command}" if self.command else ""
        return f"{head}{note}\n출력:\n{body}"


def blocked_reason(command: str) -> str | None:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"안전을 위해 허용하지 않는 명령입니다 (규칙: {pattern})."
    return None


def rewrite_command(command: str) -> str:
    """가상환경에 pip 가 없으므로 pip install 을 uv pip install 로 바꾼다."""
    if PIP_INSTALL.search(command) and shutil.which("uv"):
        return PIP_INSTALL.sub("uv pip install", command, count=1)
    return command


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    venv = Path(sys.prefix)
    bin_dir = Path(sys.executable).parent
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv)  # uv pip install 대상
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONHOME", None)
    return env


def run_shell(command: str, cwd: Path, timeout: int = 180) -> ShellResult:
    command = command.strip()
    if not command:
        return ShellResult(ok=False, exit_code=None, output="빈 명령입니다.", command="", blocked=True)
    reason = blocked_reason(command)
    if reason:
        return ShellResult(ok=False, exit_code=None, output=reason, command=command, blocked=True)

    actual = rewrite_command(command)
    cwd = cwd.resolve()
    cwd.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            actual,
            shell=True,
            cwd=cwd,
            env=_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        partial = _decode(e.output or b"")
        return ShellResult(
            ok=False, exit_code=None, output=f"{partial}\n실행 시간이 {timeout}초를 넘어 중단했습니다.", command=actual
        )

    output = _decode(proc.stdout)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:2000] + f"\n...(중간 {len(output) - MAX_OUTPUT_CHARS}자 생략)...\n" + output[-(MAX_OUTPUT_CHARS - 2000) :]
    return ShellResult(ok=proc.returncode == 0, exit_code=proc.returncode, output=output, command=actual)


def environment_note() -> str:
    return f"{platform.system()} {platform.release()}, 셸: {SHELL_NAME}, Python {platform.python_version()}"
