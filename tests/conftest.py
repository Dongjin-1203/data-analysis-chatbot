import os
import tempfile

# app 모듈을 import 하기 전에 테스트 전용 저장소를 지정한다
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="chatbot-test-")
