# 필요한 파이썬 라이브러리 목록
required_libraries=("nest_asyncio" "llama_index" "langchain" "openai")

# 설치되지 않은 라이브러리를 저장할 배열
missing_libraries=()

# 필요한 라이브러리가 설치되어 있는지 확인
for library in "${required_libraries[@]}"; do
    if ! python -c "import $library" &> /dev/null; then
        missing_libraries+=("$library")
    fi
done

# 설치되지 않은 라이브러리가 있을 경우 설치
if [ ${#missing_libraries[@]} -gt 0 ]; then
    echo "다음 라이브러리가 설치되지 않았습니다: ${missing_libraries[@]}"
    echo "라이브러리를 설치합니다..."
    
    pip install "${missing_libraries[@]}"
    
    echo "라이브러리 설치가 완료되었습니다."
else
    echo "모든 필수 라이브러리가 설치되어 있습니다."
fi
echo "코드 실행 중"
# read_doc.py 실행
python read_doc.py "$1" "$2" "$3"