"""에이전트가 작성한 Pandas 코드를 별도 Python 프로세스에서 실행한다."""

import json
import locale
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "sandbox_runner.py"


@dataclass
class ExecutionResult:
    ok: bool
    output: str = ""
    error: str | None = None
    images: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """에이전트에게 돌려줄 텍스트."""
        parts = ["[실행 성공]" if self.ok else "[실행 실패]"]
        if self.output.strip():
            parts.append(f"출력:\n{self.output.rstrip()}")
        elif self.ok:
            parts.append("출력 없음 (결과를 보려면 print()를 사용하거나 마지막 줄에 값을 두세요)")
        if self.error:
            parts.append(f"오류: {self.error}")
        if self.images:
            parts.append(f"그래프 {len(self.images)}개가 생성되어 사용자 화면에 표시됩니다.")
        return "\n".join(parts)


def _decode(raw: bytes) -> str:
    """하위 프로세스의 stderr 는 OS 기본 인코딩(Windows: cp949)일 수 있다."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def run_code(code: str, data_path: str | None, out_dir: Path, timeout: int = 30) -> ExecutionResult:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if data_path:
        # 이전 버전에서 상대 경로로 저장된 세션도 서버 기준 절대 경로로 바꿔 전달한다
        data_path = str(Path(data_path).resolve())
        if not Path(data_path).is_file():
            return ExecutionResult(
                ok=False,
                error=f"업로드된 데이터 파일을 찾을 수 없습니다: {Path(data_path).name}. 파일을 다시 첨부해 주세요.",
            )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(RUNNER), data_path or "", str(out_dir)],
            input=code.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            cwd=out_dir,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(ok=False, error=f"실행 시간이 {timeout}초를 넘어 중단했습니다.")

    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    last_line = stdout.rsplit("\n", 1)[-1] if stdout else ""
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        stderr = _decode(proc.stderr).strip()
        return ExecutionResult(ok=False, error=f"실행기 오류: {stderr[-2000:] or stdout[-2000:] or '알 수 없음'}")
    return ExecutionResult(**data)
