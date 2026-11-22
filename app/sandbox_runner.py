"""별도 프로세스에서 Pandas 코드를 실행하는 러너.

app.executor 가 `python -I sandbox_runner.py <data_path> <out_dir>` 로 실행하고,
실행할 코드는 stdin 으로 전달한다. 결과는 마지막 줄에 JSON 으로 출력한다.

- `df` (업로드된 데이터), `pd`, `np`, `plt`, `DATA_PATH` 가 미리 준비된다.
- 마지막 줄이 식(expression)이면 노트북처럼 그 값을 출력한다.
- 열려 있는 matplotlib 그림은 out_dir 에 PNG 로 저장한다.

주의: 별도 프로세스·시간 제한으로 서버를 보호할 뿐, 완전한 보안 샌드박스는 아니다.
"""

import ast
import io
import json
import sys
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

MAX_OUTPUT_CHARS = 8000


def load_data(path: str):
    import pandas as pd

    p = Path(path)
    if p.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "cp949"):
            try:
                return pd.read_csv(p, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(p)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def run(code: str, data_path: str | None, out_dir: Path) -> dict:
    import numpy as np
    import pandas as pd

    plt = setup_matplotlib()
    # 모델이 답변 예시처럼 pd.read_csv(CSV_PATH) 를 쓰더라도 동작하도록 CSV_PATH 도 제공한다
    namespace = {"pd": pd, "np": np, "plt": plt, "DATA_PATH": data_path, "CSV_PATH": data_path, "__name__": "__main__"}
    if data_path:
        try:
            namespace["df"] = load_data(data_path)
        except Exception as e:
            return {"ok": False, "output": "", "error": f"데이터 파일을 불러오지 못했습니다: {type(e).__name__}: {e}", "images": []}

    buf = io.StringIO()
    ok, error = True, None
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            tree = ast.parse(code, mode="exec")
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<agent_code>", "exec"), namespace)
            if last_expr is not None:
                value = eval(compile(last_expr, "<agent_code>", "eval"), namespace)
                if value is not None:
                    print(value.to_string() if isinstance(value, (pd.DataFrame, pd.Series)) else repr(value))
        except Exception as e:  # 에이전트가 원인을 설명할 수 있도록 실제 오류를 그대로 돌려준다
            ok = False
            tb = traceback.extract_tb(e.__traceback__)
            line = next((f.lineno for f in reversed(tb) if f.filename == "<agent_code>"), None)
            error = f"{type(e).__name__}: {e}" + (f" (코드 {line}번째 줄)" if line else "")

    images = []
    for num in plt.get_fignums():
        name = f"{uuid.uuid4().hex[:12]}.png"
        plt.figure(num).savefig(out_dir / name, dpi=110, bbox_inches="tight")
        images.append(name)
    plt.close("all")

    output = buf.getvalue()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n...(출력이 길어 {len(output) - MAX_OUTPUT_CHARS}자 생략됨)"
    return {"ok": ok, "output": output, "error": error, "images": images}


def main() -> None:
    data_path = sys.argv[1] or None
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    code = sys.stdin.buffer.read().decode("utf-8")
    result = run(code, data_path, out_dir)
    # Windows 콘솔 인코딩(cp949)과 무관하게 UTF-8 로 내보낸다
    sys.stdout.buffer.write(("\n" + json.dumps(result, ensure_ascii=False)).encode("utf-8"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
