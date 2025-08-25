# fastapi_server.py
# Hybrid RAG + Pricing API (MD/CSV + Azure Prices)
# - /debug/db, /diag/vm, /categories, /regions, /config, /fx
# - /options, /items  (NULL/NaN/UoM 방어 + 행단위 try/except + JSON sanitize)
# - /answer (RAG) 기존 유지 (임베딩/LLM 환경변수 없을 땐 우회)
# - 전역 예외 핸들러: 500도 항상 JSON(detail+traceback)

import os, re, sqlite3, traceback, math
from typing import Dict, Any, Optional, Literal, List, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ==== (선택) RAG 관련 - 환경이 없으면 LLM 없이 동작 ====
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.schema import Document

app = FastAPI(title="Hybrid RAG + Pricing API (MD/CSV + Azure Prices)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- 전역 예외 핸들러: 항상 JSON으로 반환 --------
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("[UNHANDLED]", exc)   # 콘솔 로그
    print(tb)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}", "traceback": tb}
    )

# --------------------- RAG ---------------------
AZURE_API_KEY       = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT_BASE = (os.environ.get("AZURE_OPENAI_ENDPOINT", "") or "").rstrip("/")
OPENAI_API_VERSION  = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_CHAT_DEPLOY   = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
AZURE_EMBED_DEPLOY  = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# 환경이 없으면 우회 모드로
USE_LLM   = bool(AZURE_API_KEY and AZURE_ENDPOINT_BASE)
USE_EMBED = bool(AZURE_API_KEY and AZURE_ENDPOINT_BASE)

def _maybe_llm():
    if not USE_LLM:
        return None
    return AzureChatOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT_BASE,
        azure_deployment=AZURE_CHAT_DEPLOY,
        openai_api_version=OPENAI_API_VERSION,
        temperature=0.2, max_tokens=1500, request_timeout=60,
    )
azure_llm = _maybe_llm()

# 임베딩/벡터스토어는 생성 시도하되, USE_EMBED=False면 실제 검색은 스킵
embedding_model = AzureOpenAIEmbeddings(
    api_key=AZURE_API_KEY or "nokey",
    azure_endpoint=AZURE_ENDPOINT_BASE or "http://localhost",
    azure_deployment=AZURE_EMBED_DEPLOY,
    openai_api_version=OPENAI_API_VERSION,
)

PERSIST_DIR  = os.path.join(BASE_DIR, "chromaDB")
COLLECTION   = "langchain"
VECTORSTORE  = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION,
)

chats_by_session_id: Dict[str, InMemoryChatMessageHistory] = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chats_by_session_id:
        chats_by_session_id[session_id] = InMemoryChatMessageHistory()
    return chats_by_session_id[session_id]

class QARequest(BaseModel):
    question: str
    session_id: str
    target_kb: Optional[Literal["auto", "md", "csv", "all"]] = "auto"
    k: int = 6
    mode: Literal["rag_strict", "rag_assist", "llm_only"] = "rag_assist"
    detail: Literal["concise", "standard", "rich"] = "rich"
    max_tokens: int = 1500
    include_examples: bool = True
    include_table: bool = True

class QAResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def join_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)

def docs_to_refs(docs: List[Document]) -> List[Dict[str, str]]:
    file_set, link_set, refs = set(), set(), []
    for d in docs:
        meta = d.metadata or {}
        url = meta.get("url"); src = meta.get("source")
        if url: link_set.add(url.strip())
        elif src: file_set.add(os.path.basename(src))
    for f in sorted(file_set): refs.append({"출처": "파일", "값": f})
    for u in sorted(link_set): refs.append({"출처": "링크", "값": u})
    return refs

def _structure_instructions(detail: str, include_examples: bool, include_table: bool) -> str:
    length_rule = {"concise": "- 4~6문장.\n", "standard": "- 8~12문장.\n"}.get(detail, "- 12~20문장.\n")
    parts = ["1) 서비스 요약", "2) 핵심 개념/정의", "3) 주요 기능/옵션(불릿)", "4) 다음 단계/추가 자료"]
    return "표현 규칙:\n"+length_rule+"- 한국어로 명확하게.\n"+ "\n".join(f"- {p}" for p in parts)

def build_prompt_rag(context: str, question: str, history_text: str, req: QARequest) -> str:
    header = "다음은 문맥입니다:\n" + context + "\n\n"
    history = f"이전 대화:\n{history_text}\n\n" if history_text else ""
    structure = _structure_instructions(req.detail, req.include_examples, req.include_table)
    return f"{history}{header}{structure}\n질문: {question}\n\n구조화 응답."

def build_prompt_llm_only(question: str, history_text: str, req: QARequest) -> str:
    history = f"이전 대화:\n{history_text}\n\n" if history_text else ""
    structure = _structure_instructions(req.detail, req.include_examples, req.include_table)
    return f"{history}{structure}\n질문: {question}\n\n구조화 응답."

def call_llm_with_fallback(prompt: str, max_tokens: int) -> str:
    if not USE_LLM:
        # LLM 비활성화 시 에러 내지 말고 안내 문구 반환
        return "LLM 비활성화(환경변수 미설정) 상태입니다. 검색 없이 안내만 제공합니다."
    try:
        return azure_llm.invoke(prompt, max_tokens=max_tokens).content
    except TypeError:
        return azure_llm.invoke(prompt).content

def simple_route(question: str) -> Literal["md", "csv", "all"]:
    q = question.lower()
    if any(k in q for k in ["가이드","튜토리얼","사용법"]): return "md"
    if any(k in q for k in ["가격","요금","price","vm","database","storage"]): return "csv"
    return "all"

def search_with_scores(kb: Optional[str], query: str, k: int) -> List[Tuple[Document, float]]:
    # 임베딩 자격증명 없으면 검색 스킵 → 빈 리스트
    if not USE_EMBED:
        return []
    filt = {"kb": f"kb_{kb}"} if kb in ("md", "csv") else None
    return VECTORSTORE.similarity_search_with_score(query, k=k, filter=filt)

@app.get("/")
def health():
    return {"message": "Hybrid RAG/LLM + Pricing OK",
            "persist": PERSIST_DIR, "collection": COLLECTION,
            "api_version": OPENAI_API_VERSION,
            "use_llm": USE_LLM, "use_embed": USE_EMBED}

@app.post("/answer", response_model=QAResponse)
def answer_question(req: QARequest):
    chat_history = get_chat_history(req.session_id)
    history_text = "\n".join([m.content for m in chat_history.messages])

    # LLM-only 강제 모드
    if req.mode == "llm_only":
        result = call_llm_with_fallback(build_prompt_llm_only(req.question, history_text, req), req.max_tokens)
        chat_history.add_user_message(req.question); chat_history.add_ai_message(result)
        return QAResponse(answer=result, references=[], metadata={"mode": req.mode})

    # RAG assist/strict 모드
    kb = req.target_kb if req.target_kb and req.target_kb != "auto" else simple_route(req.question)
    if kb == "all":
        pairs = search_with_scores("md", req.question, req.k) + search_with_scores("csv", req.question, req.k)
    else:
        pairs = search_with_scores(kb, req.question, req.k)
    docs = [d for d, _ in pairs]

    if req.mode == "rag_strict" and not docs:
        result = "문맥에서 관련 내용을 찾지 못했습니다. 질문을 더 구체화하거나 다른 KB를 지정해 보세요."
        return QAResponse(answer=result, references=[], metadata={"mode": req.mode, "final_kb_used": kb, "retrieved_docs_count": 0})

    prompt = build_prompt_rag(join_docs(docs), req.question, history_text, req) if docs else build_prompt_llm_only(req.question, history_text, req)
    result = call_llm_with_fallback(prompt, req.max_tokens)
    chat_history.add_user_message(req.question); chat_history.add_ai_message(result)
    return QAResponse(answer=result, references=docs_to_refs(docs),
                      metadata={"mode": req.mode, "final_kb_used": kb, "retrieved_docs_count": len(docs)})

# ------------------- Pricing API ----------------------

DB_PATH = os.path.join(BASE_DIR, "data", "azure_price_data.db")
DB_PATH = os.environ.get("PRICE_DB", DB_PATH)

TABLE_VM   = "azure_vm_prices_filtered_specs_koreacentral"
TABLE_DISK = "azure_disk_prices_filtered_specs_koreacentral"
TABLE_DB   = "filtered_db_compute_prices"
TABLE_DBR  = "0821_databricks_prices_koreacentral"
TABLE_AOAI = "0821_aoai_prices_koreacentral"

def _conn():
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 연결 실패: {e}")

# 테이블 식별자 쿼팅 (SQLite)
def qt(name: str) -> str:
    return name if (name.startswith('"') and name.endswith('"')) else f'"{name}"'

@app.get("/debug/db")
def debug_db():
    exists = os.path.exists(DB_PATH)
    tables = []
    if exists:
        try:
            con = sqlite3.connect(DB_PATH); cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            con.close()
        except Exception as e:
            tables = [f"error: {e}"]
    return {"db_path": DB_PATH, "exists": exists, "tables": tables}

# ---- JSON sanitize: NaN/Inf → null ----
def _json_sanitize(obj):
    """JSON 직렬화 전에 NaN/Inf/넘파이 스칼라 등을 안전하게 변환."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):  # NaN/Inf/-Inf
            return None
        return obj
    if hasattr(obj, "item"):  # numpy/pandas scalar
        try:
            return _json_sanitize(obj.item())
        except Exception:
            return None
    return obj

# ---- 공통 유틸 ----
def _fmt_scalar(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    try:
        fv = float(v)
        return str(int(fv)) if fv.is_integer() else str(fv)
    except Exception:
        return str(v)

def safe_float(v, default=None):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default

def safe_uom(v: Optional[str]) -> str:
    return (v or "").strip()

@app.get("/categories", response_model=List[str])
def categories():
    return ["VM", "Disk", "DB", "Databricks", "AOAI"]

@app.get("/regions", response_model=List[str])
def regions():
    return ["koreacentral"]

@app.get("/config")
def get_config():
    return {"vat_rate": 0.10}

@app.get("/fx")
def get_fx(base: str = "USD", target: str = "KRW"):
    rate = float(os.environ.get("FX_USD_KRW", "1390.0"))
    if base == "USD" and target == "KRW":
        return {"base": base, "target": target, "rate": rate}
    if base == "KRW" and target == "USD":
        return {"base": base, "target": 1.0 / rate, "rate": 1.0 / rate}
    return {"base": base, "target": target, "rate": 1.0}

# ---- 옵션 조회 ----
@app.get("/options")
def options(category: str = Query(...), region: Optional[str] = None):
    conn = _conn()
    try:
        if category == "VM":
            q = f"SELECT DISTINCT OS, Cores, MemoryGB, ReservationTerm FROM {qt(TABLE_VM)} WHERE Location='koreacentral'"
            df = pd.read_sql_query(q, conn)
            return {
                "OS": sorted([x for x in df["OS"].dropna().unique().tolist() if str(x).strip()]),
                "Cores": sorted([int(x) for x in df["Cores"].dropna().unique().tolist()]),
                "MemoryGB": sorted([int(x) for x in df["MemoryGB"].dropna().unique().tolist()]),
                "ReservationTerm": sorted([str(x) for x in df["ReservationTerm"].dropna().unique().tolist()])
            }

        if category == "Disk":
            q = f"SELECT DISTINCT DiskCategory, Tier, Provisioned_GiB, Provisioned_IOPS, ReservationTerm FROM {qt(TABLE_DISK)} WHERE Location='koreacentral'"
            df = pd.read_sql_query(q, conn)
            return {
                "DiskCategory": sorted([x for x in df["DiskCategory"].dropna().unique().tolist() if str(x).strip()]),
                "Tier": sorted([x for x in df["Tier"].dropna().unique().tolist() if str(x).strip()]),
                "Provisioned_GiB": sorted([int(x) for x in df["Provisioned_GiB"].dropna().unique().tolist()]),
                "Provisioned_IOPS": sorted([int(x) for x in df["Provisioned_IOPS"].dropna().unique().tolist()]),
                "ReservationTerm": sorted([str(x) for x in df["ReservationTerm"].dropna().unique().tolist()])
            }

        if category == "DB":
            q = f"SELECT DISTINCT serviceName, skuName, reservationTerm FROM {qt(TABLE_DB)} WHERE armRegionName='koreacentral'"
            df = pd.read_sql_query(q, conn)
            return {
                "serviceName": sorted([x for x in df["serviceName"].dropna().unique().tolist() if str(x).strip()]),
                "skuName": sorted([x for x in df["skuName"].dropna().unique().tolist() if str(x).strip()]),
                "ReservationTerm": sorted([str(x) for x in df["reservationTerm"].dropna().unique().tolist()])
            }

        if category == "AOAI":
            q = f"SELECT DISTINCT metername FROM {qt(TABLE_AOAI)}"
            df = pd.read_sql_query(q, conn)
            meters = [x for x in df["metername"].dropna().astype(str).map(str.strip).tolist() if x]
            return {"metername": sorted(meters)}

        if category == "Databricks":
            q = f"SELECT DISTINCT metername FROM {qt(TABLE_DBR)}"
            df = pd.read_sql_query(q, conn)
            meters = [x for x in df["metername"].dropna().astype(str).map(str.strip).tolist() if x]
            return {"metername": sorted(meters)}

        raise HTTPException(status_code=400, detail="unknown category")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"/options error: {e}\n{tb}")
    finally:
        try: conn.close()
        except: pass

# ---- 아이템 조회 ----
@app.get("/items")
def items(
    category: str = Query(...),
    region: Optional[str] = Query(None),

    # VM
    OS: Optional[str] = None,
    Cores: Optional[int] = None,
    MemoryGB: Optional[int] = None,
    VM_ReservationTerm: Optional[str] = None,

    # Disk
    DiskCategory: Optional[str] = None,
    Tier: Optional[str] = None,
    Provisioned_GiB: Optional[int] = None,
    Provisioned_IOPS: Optional[int] = None,
    Disk_ReservationTerm: Optional[str] = None,

    # DB
    serviceName: Optional[str] = None,
    skuName: Optional[str] = None,
    DB_ReservationTerm: Optional[str] = None,

    # Databricks / AOAI
    productname: Optional[str] = None,
    metername: Optional[str] = None,
):
    conn = _conn()
    try:
        rows: List[Dict[str, Any]] = []

        # ---------- VM ----------
        if category == "VM":
            q = f"SELECT * FROM {qt(TABLE_VM)} WHERE Location='koreacentral'"
            cond, args = [], []
            if OS: cond += ["OS=?"]; args += [OS]
            if Cores is not None: cond += ["Cores=?"]; args += [Cores]
            if MemoryGB is not None: cond += ["MemoryGB=?"]; args += [MemoryGB]
            if VM_ReservationTerm: cond += ["ReservationTerm=?"]; args += [VM_ReservationTerm]
            if cond: q += " AND " + " AND ".join(cond)

            df = pd.read_sql_query(q, conn, params=args)

            for idx, r in df.iterrows():
                try:
                    price_hour = safe_float(r.get("RetailPrice"))
                    if price_hour is None:
                        continue
                    uom = safe_uom(r.get("UnitOfMeasure"))
                    ppm = price_hour * 730 if uom.lower() in ("1 hour","hour","per hour","hourly") else price_hour

                    row = {
                        "category": "VM",
                        "name": r.get("SKUName"),
                        "region": r.get("Location"),
                        "reservation_term": r.get("ReservationTerm") or "",
                        "unit_of_measure": uom,
                        "retail_price": price_hour,
                        "price_per_month": ppm,
                        "price_per_month_after_rule": ppm,  # 할인 없음
                        "attributes": {
                            "OS": r.get("OS"),
                            "Cores": r.get("Cores"),
                            "MemoryGB": r.get("MemoryGB"),
                        },
                        "label": f"{r.get('SKUName')} | {r.get('Location')} | {r.get('OS')} | {_fmt_scalar(r.get('Cores'))}vCPU | {_fmt_scalar(r.get('MemoryGB'))}GB"
                    }
                    rows.append(_json_sanitize(row))
                except Exception as e:
                    print(f"[WARN][VM] skip row #{idx}: {e}")
            return rows  # 빈 리스트여도 200

        # ---------- Disk ----------
        elif category == "Disk":
            q = f"SELECT * FROM {qt(TABLE_DISK)} WHERE Location='koreacentral'"
            cond, args = [], []
            if DiskCategory: cond += ["DiskCategory=?"]; args += [DiskCategory]
            if Tier: cond += ["Tier=?"]; args += [Tier]
            if Provisioned_GiB is not None: cond += ["Provisioned_GiB=?"]; args += [Provisioned_GiB]
            if Provisioned_IOPS is not None: cond += ["Provisioned_IOPS=?"]; args += [Provisioned_IOPS]
            if Disk_ReservationTerm: cond += ["ReservationTerm=?"]; args += [Disk_ReservationTerm]
            if cond: q += " AND " + " AND ".join(cond)

            df = pd.read_sql_query(q, conn, params=args)
            for idx, r in df.iterrows():
                try:
                    price = safe_float(r.get("RetailPrice"))
                    if price is None:
                        continue
                    uom = safe_uom(r.get("UnitOfMeasure"))
                    ppm = price * 730 if uom.lower() in ("1 hour","hour","per hour","hourly") else price

                    row = {
                        "category": "Disk",
                        "name": r.get("SKUName_Retail"),
                        "region": r.get("Location"),
                        "reservation_term": r.get("ReservationTerm") or "",
                        "unit_of_measure": uom,
                        "retail_price": price,
                        "price_per_month": ppm,
                        "price_per_month_after_rule": ppm,
                        "attributes": {
                            "DiskCategory": r.get("DiskCategory"),
                            "Tier": r.get("Tier"),
                            "Provisioned_GiB": r.get("Provisioned_GiB"),
                            "Provisioned_IOPS": r.get("Provisioned_IOPS"),
                        },
                        "label": f"{r.get('SKUName_Retail')} | {r.get('Location')} | {r.get('Tier')} | {_fmt_scalar(r.get('Provisioned_GiB'))}GiB"
                    }
                    rows.append(_json_sanitize(row))
                except Exception as e:
                    print(f"[WARN][Disk] skip row #{idx}: {e}")
            return rows

        # ---------- DB ----------
        elif category == "DB":
            q = f"SELECT * FROM {qt(TABLE_DB)} WHERE armRegionName='koreacentral'"
            cond, args = [], []
            if serviceName: cond += ["serviceName=?"]; args += [serviceName]
            if skuName: cond += ["skuName=?"]; args += [skuName]
            if DB_ReservationTerm: cond += ["reservationTerm=?"]; args += [DB_ReservationTerm]
            if cond: q += " AND " + " AND ".join(cond)

            df = pd.read_sql_query(q, conn, params=args)
            for idx, r in df.iterrows():
                try:
                    price = safe_float(r.get("retailPrice"))
                    if price is None:
                        continue
                    uom = safe_uom(r.get("unitOfMeasure"))
                    ppm = price  # 그대로 사용

                    row = {
                        "category": "DB",
                        "name": r.get("serviceName"),
                        "region": r.get("armRegionName"),
                        "reservation_term": r.get("reservationTerm") or "",
                        "unit_of_measure": uom,
                        "retail_price": price,
                        "price_per_month": ppm,
                        "price_per_month_after_rule": ppm,
                        "attributes": {
                            "serviceName": r.get("serviceName"),
                            "skuName": r.get("skuName"),
                        },
                        "label": f"{r.get('serviceName')} | {r.get('skuName')} | {r.get('armRegionName')}"
                    }
                    rows.append(_json_sanitize(row))
                except Exception as e:
                    print(f"[WARN][DB] skip row #{idx}: {e}")
            return rows

        # ---------- Databricks ----------
        elif category == "Databricks":
            q = f"SELECT * FROM {qt(TABLE_DBR)}"
            cond, args = [], []
            if metername: cond += ["metername=?"]; args += [metername]
            if cond: q += " WHERE " + " AND ".join(cond)

            df = pd.read_sql_query(q, conn, params=args)
            for idx, r in df.iterrows():
                try:
                    price = safe_float(r.get("retailprice"))
                    if price is None:
                        continue
                    row = {
                        "category": "Databricks",
                        "name": r.get("metername"),
                        "region": "koreacentral",
                        "reservation_term": "",
                        "unit_of_measure": safe_uom(r.get("unitofmeasure")),
                        "retail_price": price,
                        "price_per_month": price,                # 프론트에서 월 환산(730h) * 개월 적용
                        "price_per_month_after_rule": price,
                        "attributes": {"metername": r.get("metername")},
                        "label": f"{r.get('metername')}"
                    }
                    rows.append(_json_sanitize(row))
                except Exception as e:
                    print(f"[WARN][Databricks] skip row #{idx}: {e}")
            return rows

        # ---------- AOAI ----------
        elif category == "AOAI":
            q = f"SELECT * FROM {qt(TABLE_AOAI)}"
            cond, args = [], []
            if productname: cond += ["productname=?"]; args += [productname]
            if metername: cond += ["metername=?"]; args += [metername]
            if cond: q += " WHERE " + " AND ".join(cond)

            df = pd.read_sql_query(q, conn, params=args)
            for idx, r in df.iterrows():
                try:
                    price = safe_float(r.get("retailprice"))
                    if price is None:
                        continue
                    row = {
                        "category": "AOAI",
                        "name": r.get("metername"),
                        "region": "koreacentral",
                        "reservation_term": "",
                        "unit_of_measure": safe_uom(r.get("unitofmeasure")),  # 천 토큰 등 표시용
                        "retail_price": price,
                        "price_per_month": price,               # 프론트에서 (in/out 토큰 수, /1000) 등 반영
                        "price_per_month_after_rule": price,
                        "attributes": {
                            "productname": r.get("productname"),
                            "metername": r.get("metername"),
                        },
                        "label": f"{r.get('productname')} | {r.get('metername')}"
                    }
                    rows.append(_json_sanitize(row))
                except Exception as e:
                    print(f"[WARN][AOAI] skip row #{idx}: {e}")
            return rows

        else:
            raise HTTPException(status_code=400, detail="unknown category")

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"/items error: {e}\n{tb}")
    finally:
        try: conn.close()
        except: pass

# ---- 진단: VM 통계 (문제 데이터 빠르게 확인) ----
@app.get("/diag/vm")
def diag_vm():
    conn = _conn()
    try:
        q = f"SELECT RetailPrice, UnitOfMeasure, SKUName, Location, OS, Cores, MemoryGB FROM {qt(TABLE_VM)} WHERE Location='koreacentral'"
        df = pd.read_sql_query(q, conn)
        total = len(df)
        null_price = int(df['RetailPrice'].isna().sum()) if 'RetailPrice' in df.columns else -1
        sample_null = df[df['RetailPrice'].isna()].head(5).to_dict(orient="records") if 'RetailPrice' in df.columns else []
        uoms = sorted([str(x) for x in df['UnitOfMeasure'].dropna().unique().tolist()]) if 'UnitOfMeasure' in df.columns else []
        return {"total": total, "null_price": null_price, "unit_of_measures": uoms, "sample_null": sample_null}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"/diag/vm error: {e}\n{tb}")
    finally:
        try: conn.close()
        except: pass
