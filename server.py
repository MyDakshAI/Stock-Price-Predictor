"""
server.py

Custom web dashboard for the stock direction predictor.
Run with:  uvicorn server:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analysis import analyze_compare, analyze_single, analyze_walk_forward

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Stock Direction Lab", version="1.0")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=80)
    years: int = Field(5, ge=2, le=10)
    threshold: float = Field(0.50, ge=0.45, le=0.65)
    use_market: bool = True
    walk_forward: bool = False
    folds: int = Field(5, ge=3, le=8)


def parse_tickers(raw: str) -> list[str]:
    tickers = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="Enter at least one ticker.")
    return tickers


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    tickers = parse_tickers(req.ticker)
    try:
        if len(tickers) > 1:
            return analyze_compare(
                tickers, req.years, req.threshold, req.use_market, req.walk_forward
            )
        if req.walk_forward:
            return analyze_walk_forward(
                tickers[0], req.years, req.threshold, req.use_market, req.folds
            )
        return analyze_single(tickers[0], req.years, req.threshold, req.use_market)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        message = str(exc)
        if "No data returned" in message:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No price data came back for that ticker. Check the symbol "
                    "(for example AAPL, not Apple), or try again if Yahoo Finance "
                    "is rate-limiting requests."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
