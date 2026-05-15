from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

try:
    from fredapi import Fred
except Exception:
    Fred = None

load_dotenv()

ASSETS: Dict[str, Dict] = {
    "S&P 500": {"ticker":"SPY","model":"equity_us","dd":[-0.05,-0.10,-0.15],"er":7.0,"vp":1.0,"desc":"US large-cap equities: earnings, valuation, rates, credit, breadth, mega-cap leadership."},
    "Gold": {"ticker":"GLD","model":"gold","dd":[-0.07,-0.12,-0.20],"er":3.0,"vp":0.0,"desc":"Gold: real yields, dollar, inflation, safe-haven demand, equity stress."},
    "Oil": {"ticker":"USO","model":"oil","dd":[-0.10,-0.20,-0.30],"er":4.0,"vp":0.0,"desc":"Oil: supply risk, demand risk, dollar, global growth, inventories, OPEC."},
    "Bitcoin": {"ticker":"BTC-USD","model":"bitcoin","dd":[-0.15,-0.25,-0.35],"er":12.0,"vp":0.0,"desc":"Bitcoin: liquidity, real yields, dollar, risk appetite, flows, leverage, momentum."},
    "FTSE All-World": {"ticker":"VWRL.L","model":"equity_global","dd":[-0.06,-0.12,-0.20],"er":6.5,"vp":0.5,"desc":"Global equities: US concentration, global growth, rates, dollar, credit."},
    "FTSE Emerging Markets": {"ticker":"VFEM.L","model":"equity_em","dd":[-0.08,-0.15,-0.25],"er":7.5,"vp":0.0,"desc":"Emerging markets: dollar, US real yields, China/global growth, EM FX, commodities."},
}

AUX = {
    "S&P 500":"SPY", "S&P 500 Index":"^GSPC", "Equal-weight S&P 500":"RSP",
    "Nasdaq 100":"QQQ", "Russell 2000":"IWM", "VIX":"^VIX", "High Yield ETF":"HYG",
    "Investment Grade ETF":"LQD", "Utilities":"XLU", "Consumer Staples":"XLP", "Industrials":"XLI",
    "Financials":"XLF", "Energy":"XLE", "Semiconductors":"SMH", "US Dollar ETF":"UUP",
    "WTI Futures":"CL=F", "Brent Futures":"BZ=F", "Oil ETF":"USO", "Gold ETF":"GLD",
    "Gold Futures":"GC=F", "Bitcoin":"BTC-USD", "FTSE All-World":"VWRL.L",
    "FTSE All-World Acc":"VWRP.L", "FTSE Emerging Markets":"VFEM.L", "FTSE Emerging Markets Alt":"VFEG.L",
    "China ETF":"FXI", "Emerging Markets ETF":"EEM", "World ex-US ETF":"VEU",
}

FRED_SERIES = {
    "10Y Treasury Yield":"DGS10", "2Y Treasury Yield":"DGS2", "10Y Real Yield":"DFII10",
    "10Y Breakeven Inflation":"T10YIE", "High Yield Spread":"BAMLH0A0HYM2",
    "Investment Grade Spread":"BAMLC0A0CM", "Financial Conditions Index":"NFCI",
    "Initial Jobless Claims":"ICSA", "Trade Weighted Dollar":"DTWEXBGS",
}

@dataclass
class Signal:
    name: str
    category: str
    status: str
    score: int
    detail: str
    value: Optional[float]=None


def get_secret(name: str, default: Optional[str]=None) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        if v: return str(v)
    except Exception:
        pass
    return os.getenv(name, default)


def status(score:int)->str: return "Green" if score<=0 else "Amber" if score==1 else "Red"
def bucket(p:float)->str: return "High" if p>=0.60 else "Moderate" if p>=0.35 else "Low"
def safe(df:pd.DataFrame,c:str)->pd.Series: return df[c].dropna() if c in df else pd.Series(dtype=float)
def clamp(x,lo,hi): return max(lo,min(hi,x))

def pc(s:pd.Series,d:int)->float:
    s=s.dropna(); return np.nan if len(s)<=d else float((s.iloc[-1]/s.iloc[-d-1]-1)*100)

def dc(s:pd.Series,d:int)->float:
    s=s.dropna(); return np.nan if len(s)<=d else float(s.iloc[-1]-s.iloc[-d-1])

def corr(a:pd.Series,b:pd.Series,w:int=60)->float:
    df=pd.concat([a,b],axis=1).dropna()
    return np.nan if len(df)<w else float(df.iloc[:,0].rolling(w).corr(df.iloc[:,1]).iloc[-1])

def aggregate(sigs:List[Signal])->Tuple[int,str,float]:
    total=sum(max(s.score,0) for s in sigs); pct=total/max(2*len(sigs),1)
    return total,bucket(pct),pct

@st.cache_data(ttl=3600)
def load_market(period:str, overrides:Dict[str,str])->pd.DataFrame:
    tickers=AUX.copy(); tickers.update({k:v for k,v in overrides.items() if v})
    raw=yf.download(list(set(tickers.values())), period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    out=pd.DataFrame()
    for name,ticker in tickers.items():
        try: out[name]=raw[ticker]["Close"]
        except Exception: out[name]=np.nan
    return out.dropna(how="all").sort_index()

@st.cache_data(ttl=14400)
def load_fred(start:str)->pd.DataFrame:
    key=get_secret("FRED_API_KEY")
    if not key or Fred is None: return pd.DataFrame()
    fred=Fred(api_key=key); out=pd.DataFrame()
    for name,code in FRED_SERIES.items():
        try: out[name]=fred.get_series(code, observation_start=start)
        except Exception: out[name]=np.nan
    out.index=pd.to_datetime(out.index)
    return out.sort_index().ffill()

def fred_diag(start:str)->dict:
    key=get_secret("FRED_API_KEY")
    d={"fredapi_imported":Fred is not None,"secret_found":bool(key),"secret_length":len(key) if key else 0,"test_fetch_ok":False,"test_rows":0,"error":""}
    if not key: d["error"]="FRED_API_KEY not found in Streamlit secrets or .env."; return d
    if Fred is None: d["error"]="fredapi did not import."; return d
    try:
        s=Fred(api_key=key).get_series("DGS10", observation_start=start); d["test_rows"]=len(s); d["test_fetch_ok"]=len(s)>0
    except Exception as e: d["error"]=str(e)
    return d

# ---------- manual overlay ----------

def mscore(x:str)->int:
    return {"Positive / supportive":-1,"Neutral":0,"Negative / concerning":1,"Very negative / high risk":2,
            "Low":0,"Medium":1,"High":2,"Very high":2,"Improving":-1,"Stable / neutral":0,
            "Deteriorating":1,"Severely deteriorating":2}.get(x,0)

def manual_ui()->Dict:
    m={}
    st.subheader("Manual inputs")
    st.write("Forward-looking overlay for information FRED/yfinance do not capture well. Keep weight modest unless conviction is high.")
    c1,c2=st.columns(2)
    with c1:
        m["manual_weight"]=st.slider("Manual overlay weight",0.0,0.75,0.25,0.05)
    with c2:
        m["risk_tolerance"]=st.selectbox("Risk tolerance for action labels",["Conservative","Balanced","Aggressive"],index=1)
    st.markdown("### Macro catalysts")
    a,b,c,d=st.columns(4)
    with a:
        m["cpi"]=st.selectbox("Next CPI/PCE risk",["Low","Medium","High","Very high"],index=1)
        m["fed"]=st.selectbox("Next Fed/FOMC risk",["Low","Medium","High","Very high"],index=1)
    with b:
        m["jobs"]=st.selectbox("Payrolls/jobs risk",["Low","Medium","High","Very high"],index=1)
        m["treasury"]=st.selectbox("Treasury auction/refunding risk",["Low","Medium","High","Very high"],index=0)
    with c:
        m["days"]=st.number_input("Days to next major catalyst",0,90,14,1)
        m["sensitivity"]=st.selectbox("Market sensitivity to events",["Low","Medium","High","Very high"],index=1)
    with d:
        m["notes"]=st.text_area("Event notes",height=110)
    st.markdown("### Equities")
    a,b,c=st.columns(3)
    with a:
        m["spx_pe"]=st.number_input("S&P 500 forward P/E",5.0,40.0,21.0,0.1)
        m["spx_eps"]=st.selectbox("S&P EPS revisions",["Improving","Stable / neutral","Deteriorating","Severely deteriorating"],index=1)
    with b:
        m["spx_guidance"]=st.selectbox("S&P guidance/margins",["Positive / supportive","Neutral","Negative / concerning","Very negative / high risk"],index=1)
        m["mag7"]=st.selectbox("Mag 7 / AI earnings risk",["Low","Medium","High","Very high"],index=1)
    with c:
        m["global_eps"]=st.selectbox("Global EPS revisions",["Improving","Stable / neutral","Deteriorating","Severely deteriorating"],index=1)
        m["global_val"]=st.selectbox("Global valuation risk",["Low","Medium","High","Very high"],index=1)
    st.markdown("### Oil / geopolitics")
    a,b,c=st.columns(3)
    with a:
        m["oil_supply"]=st.selectbox("Oil supply/geopolitical risk",["Low","Medium","High","Very high"],index=1)
        m["opec"]=st.selectbox("OPEC policy risk",["Low","Medium","High","Very high"],index=1)
    with b:
        m["oil_inv"]=st.selectbox("Oil inventory trend",["Improving","Stable / neutral","Deteriorating","Severely deteriorating"],index=1)
        m["oil_demand"]=st.selectbox("Oil demand commentary",["Improving","Stable / neutral","Deteriorating","Severely deteriorating"],index=1)
    with c:
        m["geo"]=st.selectbox("General geopolitical risk",["Low","Medium","High","Very high"],index=1)
    st.markdown("### Bitcoin / EM")
    a,b,c=st.columns(3)
    with a:
        m["btc_flows"]=st.selectbox("BTC ETF flows",["Positive / supportive","Neutral","Negative / concerning","Very negative / high risk"],index=1)
        m["btc_liq"]=st.selectbox("Crypto liquidity",["Positive / supportive","Neutral","Negative / concerning","Very negative / high risk"],index=1)
    with b:
        m["btc_lev"]=st.selectbox("BTC leverage/funding",["Low","Medium","High","Very high"],index=1)
        m["china"]=st.selectbox("China macro/policy",["Positive / supportive","Neutral","Negative / concerning","Very negative / high risk"],index=1)
    with c:
        m["em_fx"]=st.selectbox("EM FX stress",["Low","Medium","High","Very high"],index=1)
        m["em_liq"]=st.selectbox("EM dollar liquidity pressure",["Low","Medium","High","Very high"],index=1)
    return m

def add_ms(sigs,name,raw,detail,w):
    weighted=raw*w; score=0 if weighted<0.5 else 1 if weighted<1.25 else 2
    sigs.append(Signal(name,"Manual overlay",status(score),score,detail,weighted))

def manual_sigs(asset:str,m:Optional[Dict])->List[Signal]:
    if not m or m.get("manual_weight",0)<=0: return []
    w=float(m.get("manual_weight",0.25)); model=ASSETS[asset]["model"]; sigs=[]
    ev=max(mscore(m.get(k,"Medium")) for k in ["cpi","fed","jobs","treasury","sensitivity"])+(2 if int(m.get("days",14))<=3 else 1 if int(m.get("days",14))<=10 else 0)
    if model in ["equity_us","equity_global","equity_em","gold","bitcoin"]:
        add_ms(sigs,"Manual event/catalyst risk",ev,f"Major catalyst in {m.get('days')} days; market sensitivity {m.get('sensitivity')}.",w)
    if model=="equity_us":
        add_ms(sigs,"Manual valuation risk",2 if m.get("spx_pe",21)>=23 else 1 if m.get("spx_pe",21)>=21 else 0,f"S&P forward P/E {m.get('spx_pe'):.1f}.",w)
        add_ms(sigs,"Manual EPS revisions",mscore(m.get("spx_eps")),f"S&P EPS revisions: {m.get('spx_eps')}.",w)
        add_ms(sigs,"Manual guidance/margins",mscore(m.get("spx_guidance")),f"S&P guidance/margins: {m.get('spx_guidance')}.",w)
        add_ms(sigs,"Manual Mag 7 / AI earnings risk",mscore(m.get("mag7")),f"Mag 7 / AI earnings risk: {m.get('mag7')}.",w)
    elif model=="equity_global":
        add_ms(sigs,"Manual global EPS revisions",mscore(m.get("global_eps")),f"Global EPS revisions: {m.get('global_eps')}.",w)
        add_ms(sigs,"Manual global valuation risk",mscore(m.get("global_val")),f"Global valuation risk: {m.get('global_val')}.",w)
        add_ms(sigs,"Manual US earnings spillover",mscore(m.get("spx_eps")),f"US EPS spillover: {m.get('spx_eps')}.",w)
    elif model=="equity_em":
        add_ms(sigs,"Manual China macro risk",mscore(m.get("china")),f"China macro/policy: {m.get('china')}.",w)
        add_ms(sigs,"Manual EM FX stress",mscore(m.get("em_fx")),f"EM FX stress: {m.get('em_fx')}.",w)
        add_ms(sigs,"Manual EM dollar liquidity",mscore(m.get("em_liq")),f"EM dollar liquidity: {m.get('em_liq')}.",w)
    elif model=="gold":
        add_ms(sigs,"Manual safe-haven/geopolitical support",-mscore(m.get("geo")),f"Geopolitical risk: {m.get('geo')}.",w)
        add_ms(sigs,"Manual inflation/Fed catalyst risk",ev,"Inflation/Fed event risk can increase gold volatility.",w)
    elif model=="oil":
        add_ms(sigs,"Manual oil supply support",-mscore(m.get("oil_supply")),f"Oil supply/geopolitical risk: {m.get('oil_supply')}.",w)
        add_ms(sigs,"Manual OPEC risk",mscore(m.get("opec")),f"OPEC policy risk: {m.get('opec')}.",w)
        add_ms(sigs,"Manual oil demand trend",mscore(m.get("oil_demand")),f"Oil demand: {m.get('oil_demand')}.",w)
        add_ms(sigs,"Manual oil inventory trend",mscore(m.get("oil_inv")),f"Oil inventories: {m.get('oil_inv')}.",w)
    elif model=="bitcoin":
        add_ms(sigs,"Manual BTC ETF flows",mscore(m.get("btc_flows")),f"BTC ETF flows: {m.get('btc_flows')}.",w)
        add_ms(sigs,"Manual crypto liquidity",mscore(m.get("btc_liq")),f"Crypto liquidity: {m.get('btc_liq')}.",w)
        add_ms(sigs,"Manual BTC leverage/funding",mscore(m.get("btc_lev")),f"BTC leverage/funding: {m.get('btc_lev')}.",w)
    return sigs

def manual_er_adj(asset:str,m:Optional[Dict])->float:
    if not m: return 0.0
    w=float(m.get("manual_weight",0.25)); model=ASSETS[asset]["model"]; adj=0.0
    if model=="equity_us": adj += -0.010*max(mscore(m.get("spx_eps")),0)-0.008*max(mscore(m.get("spx_guidance")),0)-0.005*max(mscore(m.get("mag7"))-1,0)-(.010 if m.get("spx_pe",21)>=23 else 0)
    elif model=="equity_global": adj += -0.008*max(mscore(m.get("global_eps")),0)-0.005*max(mscore(m.get("global_val"))-1,0)
    elif model=="equity_em": adj += -0.010*max(mscore(m.get("china")),0)-0.008*max(mscore(m.get("em_fx"))-1,0)-0.008*max(mscore(m.get("em_liq"))-1,0)
    elif model=="gold": adj += 0.005*max(mscore(m.get("geo"))-1,0)
    elif model=="oil": adj += 0.010*max(mscore(m.get("oil_supply"))-1,0)-0.012*max(mscore(m.get("oil_demand")),0)
    elif model=="bitcoin": adj += -0.015*max(mscore(m.get("btc_flows")),0)-0.015*max(mscore(m.get("btc_liq")),0)-0.010*max(mscore(m.get("btc_lev"))-1,0)
    return adj*w

# ---------- objective signals ----------

def trend(asset, prices):
    s=prices.dropna()
    if len(s)<220: return Signal("Trend","Drawdown risk","Amber",1,"Not enough history for 200-day trend signal.")
    latest,ma50,ma200=float(s.iloc[-1]),float(s.rolling(50).mean().iloc[-1]),float(s.rolling(200).mean().iloc[-1])
    d50,d200=(latest/ma50-1)*100,(latest/ma200-1)*100
    score=2 if latest<ma200 else 1 if latest<ma50 else 0
    return Signal("Trend","Drawdown risk",status(score),score,f"Price vs moving averages: 50d {d50:.1f}%, 200d {d200:.1f}%.",latest)

def momentum(asset, prices, highvol=False):
    r1,r3=pc(prices,21),pc(prices,63); a1,r1t,a3,r3t=(-10,-18,-18,-30) if highvol else (-4,-8,-8,-15)
    score=2 if r1<r1t or r3<r3t else 1 if r1<a1 or r3<a3 else 0
    return Signal("Momentum","Drawdown risk",status(score),score,f"1m {r1:.1f}%, 3m {r3:.1f}%.",r1)

def volsig(asset, prices, highvol=False):
    r=prices.pct_change().dropna()
    if len(r)<63: return Signal("Asset volatility","Drawdown severity","Amber",1,"Not enough volatility history.")
    v21,v63=float(r.tail(21).std()*np.sqrt(252)),float(r.tail(63).std()*np.sqrt(252)); ratio=v21/v63 if v63 else np.nan
    a,red=(0.70,1.00) if highvol else (0.28,0.40); score=2 if v21>red or ratio>1.6 else 1 if v21>a or ratio>1.3 else 0
    return Signal("Asset volatility","Drawdown severity",status(score),score,f"21d annualised vol {v21:.1%}; 21d/63d vol ratio {ratio:.2f}.",v21)

def rates(fred):
    if fred.empty or "10Y Treasury Yield" not in fred: return Signal("Rates / real yields","Macro","Amber",1,"FRED rates unavailable.")
    ten=fred["10Y Treasury Yield"].dropna(); real=fred.get("10Y Real Yield",pd.Series(dtype=float)).dropna()
    if len(ten)<30: return Signal("Rates / real yields","Macro","Amber",1,"Not enough rates history.")
    latest,chg=float(ten.iloc[-1]),dc(ten,22); rlatest=float(real.iloc[-1]) if len(real) else np.nan; rchg=dc(real,22) if len(real)>22 else np.nan
    score=2 if chg>0.40 or rlatest>2.5 or rchg>0.30 else 1 if chg>0.20 or rlatest>2.2 or rchg>0.15 else 0
    return Signal("Rates / real yields","Macro",status(score),score,f"10Y {latest:.2f}%, 1m change {chg:.2f}pp, real yield {rlatest:.2f}%.",latest)

def credit(mkt,fred):
    score=0; parts=[]; hyg,lqd=safe(mkt,"High Yield ETF"),safe(mkt,"Investment Grade ETF")
    if len(hyg)>30 and len(lqd)>30:
        r=pc((hyg/lqd).dropna(),21); parts.append(f"HYG/LQD 1m {r:.1f}%"); score=max(score,2 if r<-3 else 1 if r<-1.5 else 0)
    if not fred.empty and "High Yield Spread" in fred:
        hy=fred["High Yield Spread"].dropna()
        if len(hy)>30:
            lvl,chg=float(hy.iloc[-1]),dc(hy,22); parts.append(f"HY spread {lvl:.2f}%, 1m {chg:.2f}pp"); score=max(score,2 if lvl>5.5 or chg>0.75 else 1 if lvl>4.5 or chg>0.35 else 0)
    if not parts: return Signal("Credit stress","Macro","Amber",1,"Credit data unavailable.")
    return Signal("Credit stress","Macro",status(score),score,"; ".join(parts),None)

def dollar(mkt,fred,model):
    u=safe(mkt,"US Dollar ETF"); parts=[]; score=0; val=None
    if len(u)>30:
        r=pc(u,21); val=r; parts.append(f"UUP 1m {r:.1f}%")
        score=max(score,2 if model in ["equity_em","gold","bitcoin"] and r>3 else 1 if model in ["equity_em","gold","bitcoin"] and r>1.5 else 1 if model in ["equity_us","equity_global"] and r>3.5 else 0)
    if not fred.empty and "Trade Weighted Dollar" in fred:
        s=fred["Trade Weighted Dollar"].dropna()
        if len(s)>30: parts.append(f"FRED dollar 1m {pc(s,21):.1f}%")
    if not parts: return Signal("Dollar pressure","Macro","Amber",1,"Dollar data unavailable.")
    return Signal("Dollar pressure","Macro",status(score),score,"; ".join(parts),val)

def oil_macro(mkt,model):
    oil=safe(mkt,"WTI Futures")
    if len(oil)<30: oil=safe(mkt,"Oil ETF")
    if len(oil)<30: return Signal("Oil / commodity shock","Macro","Amber",1,"Oil data unavailable.")
    r5,r21=pc(oil,5),pc(oil,21)
    if model=="oil": score=2 if r21<-15 else 1 if r21<-8 else 0; detail=f"Oil holder trend: 5d {r5:.1f}%, 1m {r21:.1f}%."
    else: score=2 if r5>8 or r21>15 else 1 if r5>4 or r21>8 else 0; detail=f"Oil inflation-shock risk: 5d {r5:.1f}%, 1m {r21:.1f}%."
    return Signal("Oil / commodity shock","Macro",status(score),score,detail,float(oil.iloc[-1]))

def build_signals(asset, prices, mkt, fred, manual=None):
    cfg=ASSETS[asset]; model=cfg["model"]; sigs=[trend(asset,prices), momentum(asset,prices,model=="bitcoin"), volsig(asset,prices,model=="bitcoin")]
    if model in ["equity_us","equity_global","equity_em"]:
        sigs += [rates(fred), credit(mkt,fred), dollar(mkt,fred,model), oil_macro(mkt,model)]
        if asset=="S&P 500":
            sp=prices; rsp,qqq,iwm,smh=safe(mkt,"Equal-weight S&P 500"),safe(mkt,"Nasdaq 100"),safe(mkt,"Russell 2000"),safe(mkt,"Semiconductors")
            if len(rsp)>70: sigs.append(Signal("Market breadth","Latent fragility",status(2 if pc((rsp/sp).dropna(),21)<-3 and pc((rsp/sp).dropna(),63)<-5 else 1 if pc((rsp/sp).dropna(),21)<-2 or pc((rsp/sp).dropna(),63)<-3 else 0),2 if pc((rsp/sp).dropna(),21)<-3 and pc((rsp/sp).dropna(),63)<-5 else 1 if pc((rsp/sp).dropna(),21)<-2 or pc((rsp/sp).dropna(),63)<-3 else 0,f"Equal-weight vs cap-weight: 1m {pc((rsp/sp).dropna(),21):.1f}%, 3m {pc((rsp/sp).dropna(),63):.1f}%"))
            for name,series,amber,red in [("Mega-cap leadership",qqq,-1.5,-3),("Small-cap risk appetite",iwm,-3,-5),("Semiconductor leadership",smh,-3,-5)]:
                if len(series)>70:
                    r=pc((series/sp).dropna(),21); sc=2 if r<red else 1 if r<amber else 0; sigs.append(Signal(name,"Latent fragility" if "leadership" in name else "Drawdown risk",status(sc),sc,f"1m relative move {r:.1f}%",r))
        if model=="equity_global":
            sp,veu=safe(mkt,"S&P 500"),safe(mkt,"World ex-US ETF")
            if len(sp)>70 and len(veu)>70:
                r=pc((sp/veu).dropna(),63); sc=1 if r>8 else 0; sigs.append(Signal("US concentration risk","Latent fragility",status(sc),sc,f"S&P vs ex-US 3m {r:.1f}%",r))
        if model=="equity_em":
            eem,fxi,sp=safe(mkt,"Emerging Markets ETF"),safe(mkt,"China ETF"),safe(mkt,"S&P 500")
            if len(eem)>70 and len(sp)>70:
                r=pc((eem/sp).dropna(),21); sc=2 if r<-5 else 1 if r<-3 else 0; sigs.append(Signal("EM vs US risk appetite","Drawdown risk",status(sc),sc,f"EM vs S&P 1m {r:.1f}%",r))
            if len(fxi)>70:
                r=pc(fxi,21); sc=2 if r<-10 else 1 if r<-6 else 0; sigs.append(Signal("China proxy stress","Macro",status(sc),sc,f"FXI 1m {r:.1f}%",r))
    elif model=="gold":
        sigs += [dollar(mkt,fred,model)]
        real=fred.get("10Y Real Yield",pd.Series(dtype=float)).dropna() if not fred.empty else pd.Series(dtype=float)
        if len(real)>30:
            lvl,chg=float(real.iloc[-1]),dc(real,22); sc=2 if lvl>2.4 or chg>0.25 else 1 if lvl>2.0 or chg>0.12 else 0; sigs.append(Signal("Real-yield pressure","Macro",status(sc),sc,f"Real yield {lvl:.2f}%, 1m {chg:.2f}pp",lvl))
        else: sigs.append(Signal("Real-yield pressure","Macro","Amber",1,"FRED real yield unavailable."))
    elif model=="oil":
        sigs += [dollar(mkt,fred,model), oil_macro(mkt,model), credit(mkt,fred)]
        fci=fred.get("Financial Conditions Index",pd.Series(dtype=float)).dropna() if not fred.empty else pd.Series(dtype=float)
        if len(fci)>30:
            lvl,chg=float(fci.iloc[-1]),dc(fci,22); sc=2 if lvl>0.5 or chg>0.3 else 1 if lvl>0 or chg>0.15 else 0; sigs.append(Signal("Demand / financial conditions","Macro",status(sc),sc,f"FCI {lvl:.2f}, 1m {chg:.2f}",lvl))
    elif model=="bitcoin":
        sigs += [rates(fred), dollar(mkt,fred,model), credit(mkt,fred)]
        qqq=safe(mkt,"Nasdaq 100")
        if len(qqq)>70:
            c=corr(prices.pct_change(),qqq.pct_change(),60); sc=2 if c>0.55 else 1 if c>0.35 else 0; sigs.append(Signal("Risk-appetite sensitivity","Latent fragility",status(sc),sc,f"60d BTC-QQQ correlation {c:.2f}",c))
    sigs += manual_sigs(asset,manual)
    return sigs

# ---------- model ----------

def latent(sigs):
    rel=[s for s in sigs if s.category in ["Latent fragility","Manual overlay"]] or [s for s in sigs if s.name in ["Rates / real yields","Dollar pressure","Trend"]]
    pct=sum(max(s.score,0) for s in rel)/max(2*len(rel),1); lab=bucket(pct)
    return lab,pct,"Unusually sensitive to bad news/catalysts." if lab=="High" else "Some fragility present." if lab=="Moderate" else "Fragility contained."

def driver(sigs,model):
    cand=[s for s in sigs if s.status=="Red"] or [s for s in sigs if s.status=="Amber"]
    if not cand: return "Supportive / no dominant stress driver","Most monitored channels are green."
    pri={"equity_us":["Credit stress","Rates / real yields","Manual EPS revisions","Manual valuation risk","Market breadth","Mega-cap leadership","Oil / commodity shock"],"equity_global":["Credit stress","Rates / real yields","Dollar pressure","Manual global EPS revisions","US concentration risk"],"equity_em":["Dollar pressure","Rates / real yields","Manual China macro risk","Manual EM FX stress","China proxy stress","Credit stress"],"gold":["Real-yield pressure","Dollar pressure","Manual safe-haven/geopolitical support"],"oil":["Demand / financial conditions","Credit stress","Manual oil demand trend","Dollar pressure","Momentum"],"bitcoin":["Rates / real yields","Dollar pressure","Credit stress","Manual BTC ETF flows","Manual crypto liquidity","Risk-appetite sensitivity"]}.get(model,[])
    for p in pri:
        for s in cand:
            if s.name==p: return s.name,s.detail
    top=sorted(cand,key=lambda x:x.score,reverse=True)[0]; return top.name,top.detail

def hold_model(sigs,cfg,months,er_ann,cash_ann,vp_ann):
    _,rl,rp=aggregate(sigs); fl,fp,_=latent(sigs); yrs=months/12; gross=er_ann*yrs; cash=cash_ann*yrs; vp=vp_ann*yrs
    bp={"equity_us":.055,"equity_global":.050,"equity_em":.075,"gold":.040,"oil":.090,"bitcoin":.140}.get(cfg["model"],.055)
    fp_scale={"equity_us":.030,"equity_global":.025,"equity_em":.040,"gold":.020,"oil":.050,"bitcoin":.080}.get(cfg["model"],.030)
    risk_pen=rp*bp*(months/3); frag_pen=fp*fp_scale*(months/3); exp=gross-vp-risk_pen-frag_pen; excess=exp-cash
    lab="Attractive" if excess>.03 and rp<.35 else "Moderately attractive" if excess>0 and rp<.55 else "Neutral / uncertain" if excess>-.02 else "Unattractive"
    return {"label":lab,"risk_pct":rp,"frag_pct":fp,"gross":gross,"cash":cash,"valuation_penalty":vp,"risk_penalty":risk_pen,"fragility_penalty":frag_pen,"expected":exp,"excess":excess}

def action(h3,h6,m):
    tol=(m or {}).get("risk_tolerance","Balanced"); rc={"Conservative":.30,"Balanced":.40,"Aggressive":.50}.get(tol,.40); fc={"Conservative":.35,"Balanced":.45,"Aggressive":.55}.get(tol,.45)
    if h3["excess"]>.02 and h6["excess"]>.03 and h3["risk_pct"]<rc and h3["frag_pct"]<fc: return "Hold / add gradually","3–6 month expected return looks attractive versus cash and warning conditions are contained."
    if h3["excess"]>0 and h6["excess"]>0 and h3["risk_pct"]<rc+.10: return "Hold","Holding case is positive, though not exceptional."
    if h6["excess"]>0 and (h3["risk_pct"]>=rc or h3["frag_pct"]>=fc): return "Hold, avoid adding","Longer-term case may be reasonable, but near-term fragility is rising."
    if h3["excess"]<=-.01 and h3["risk_pct"]>=rc: return "Reduce exposure / raise cash","Expected return versus cash is weak and risk conditions are elevated."
    if h3["risk_pct"]>=.60: return "Hold but hedge / trim","Drawdown warning conditions are elevated."
    return "Neutral / wait for clearer signal","Signals are mixed."

def dd_probs(sigs,cfg):
    _,_,rp=aggregate(sigs); _,fp,_=latent(sigs); base={"equity_us":(.10,.04,.02),"equity_global":(.12,.05,.025),"equity_em":(.16,.08,.04),"gold":(.13,.06,.03),"oil":(.20,.12,.07),"bitcoin":(.25,.16,.10)}.get(cfg["model"],(.12,.05,.025))
    return {"mild":clamp(base[0]+.45*rp+.15*fp,.02,.90),"medium":clamp(base[1]+.30*rp+.10*fp,.01,.75),"severe":clamp(base[2]+.18*rp+.08*fp,.005,.60)}

def pctstr(x): return f"{x*100:.1f}%" if pd.notna(x) else ""

def color(v):
    colors={"Green":"#D7F7DF","Amber":"#FFF2CC","Red":"#F8D7DA"}
    return f"background-color: {colors.get(v, '')}" if v in colors else ""

# ---------- render ----------

def render_asset(asset,mkt,fred,assum,m):
    cfg=ASSETS[asset]; prices=safe(mkt,asset)
    if prices.empty: st.error(f"No price data for {asset}. Check ticker settings."); return {}
    sigs=build_signals(asset,prices,mkt,fred,m); total,rl,rp=aggregate(sigs); fl,fp,fd=latent(sigs); drv,drvdet=driver(sigs,cfg["model"]); probs=dd_probs(sigs,cfg)
    er=assum[asset]["er"]/100+manual_er_adj(asset,m); h3=hold_model(sigs,cfg,3,er,assum[asset]["cash"]/100,assum[asset]["vp"]/100); h6=hold_model(sigs,cfg,6,er,assum[asset]["cash"]/100,assum[asset]["vp"]/100); act,why=action(h3,h6,m)
    st.subheader(asset); st.caption(cfg["desc"])
    c1,c2,c3,c4=st.columns(4); c1.metric("Action view",act); c2.metric("3m attractiveness",h3["label"]); c3.metric("6m attractiveness",h6["label"]); c4.metric("Primary driver",drv)
    st.info(why); st.write(f"**Driver detail:** {drvdet}")
    c5,c6,c7,c8=st.columns(4); c5.metric("Drawdown risk",rl,f"{total}/{2*len(sigs)}"); c6.metric("Latent fragility",fl); c7.metric("3m excess vs cash",pctstr(h3["excess"])); c8.metric("6m excess vs cash",pctstr(h6["excess"]))
    dd=cfg["dd"]; d1,d2,d3=st.columns(3); d1.metric(f"Approx. {dd[0]:.0%} DD prob.",f"{probs['mild']*100:.0f}%"); d2.metric(f"Approx. {dd[1]:.0%} DD prob.",f"{probs['medium']*100:.0f}%"); d3.metric(f"Approx. {dd[2]:.0%} DD prob.",f"{probs['severe']*100:.0f}%")
    with st.expander("Holding attractiveness breakdown",expanded=True):
        bd=pd.DataFrame([{"Horizon":"3m","Gross expected":h3["gross"],"Valuation penalty":-h3["valuation_penalty"],"Risk penalty":-h3["risk_penalty"],"Fragility penalty":-h3["fragility_penalty"],"Expected return":h3["expected"],"Cash return":h3["cash"],"Excess vs cash":h3["excess"]},{"Horizon":"6m","Gross expected":h6["gross"],"Valuation penalty":-h6["valuation_penalty"],"Risk penalty":-h6["risk_penalty"],"Fragility penalty":-h6["fragility_penalty"],"Expected return":h6["expected"],"Cash return":h6["cash"],"Excess vs cash":h6["excess"]}])
        for c in bd.columns:
            if c!="Horizon": bd[c]=bd[c].map(pctstr)
        st.dataframe(bd,use_container_width=True,hide_index=True); st.caption(f"Manual expected-return adjustment: {manual_er_adj(asset,m)*100:.2f}% annualised.")
    with st.expander("Signals",expanded=True):
        st.write(f"**Latent fragility:** {fd}"); df=pd.DataFrame([s.__dict__ for s in sigs]); st.dataframe(df[["category","name","status","score","detail"]].style.applymap(color),use_container_width=True,hide_index=True)
    chart=pd.DataFrame({asset:prices,"50-day MA":prices.rolling(50).mean(),"200-day MA":prices.rolling(200).mean()}); st.plotly_chart(px.line(chart,title=f"{asset} trend"),use_container_width=True)
    return {"Asset":asset,"Action":act,"3m attractiveness":h3["label"],"6m attractiveness":h6["label"],"Drawdown risk":rl,"Latent fragility":fl,"Primary driver":drv,"3m expected return":h3["expected"],"3m excess vs cash":h3["excess"],"6m expected return":h6["expected"],"6m excess vs cash":h6["excess"],"Mild drawdown probability":probs["mild"],"Medium drawdown probability":probs["medium"],"Severe drawdown probability":probs["severe"]}

def fwd_mdd(p,idx,h):
    fut=p.iloc[idx+1:idx+h+1].dropna(); return np.nan if len(fut)==0 else float(fut.min()/p.iloc[idx]-1)

def backtest(asset,mkt,fred,horizon,sample,low,high):
    cfg=ASSETS[asset]; p=safe(mkt,asset); rows=[]; dates=p.index[260:len(p)-horizon:sample]; prog=st.progress(0,text=f"Running {asset} backtest...")
    for i,dt in enumerate(dates):
        try: sigs=build_signals(asset,p.loc[:dt],mkt.loc[:dt],fred.loc[:dt] if not fred.empty else pd.DataFrame(),None); _,_,rp=aggregate(sigs)
        except Exception: continue
        idx=p.index.get_indexer([dt],method="nearest")[0]
        if idx<0 or idx+horizon>=len(p): continue
        fwd=p.iloc[idx+horizon]/p.iloc[idx]-1; mdd=fwd_mdd(p,idx,horizon); b="Low" if rp<low else "Moderate" if rp<high else "High"; dd=cfg["dd"]
        rows.append({"Date":dt,"Risk score pct":rp,"Risk bucket":b,"Forward return":fwd,"Forward max drawdown":mdd,f"Hit {dd[0]:.0%}":mdd<=dd[0],f"Hit {dd[1]:.0%}":mdd<=dd[1],f"Hit {dd[2]:.0%}":mdd<=dd[2]})
        if i%10==0 and len(dates): prog.progress(min((i+1)/len(dates),1.0))
    prog.empty(); res=pd.DataFrame(rows)
    if res.empty: return res,pd.DataFrame()
    hits=[c for c in res.columns if c.startswith("Hit ")]; agg={"Observations":("Date","count"),"Median forward return":("Forward return","median"),"Mean forward return":("Forward return","mean"),"Worst forward return":("Forward return","min"),"Median forward drawdown":("Forward max drawdown","median"),"Worst forward drawdown":("Forward max drawdown","min")}
    for c in hits: agg[f"Probability {c.replace('Hit ','')}"]=(c,"mean")
    summ=res.groupby("Risk bucket").agg(**agg).reset_index(); summ["order"]=summ["Risk bucket"].map({"Low":0,"Moderate":1,"High":2}); return res,summ.sort_values("order").drop(columns="order")

def fmt(df):
    o=df.copy()
    for c in o.columns:
        if any(w in c.lower() for w in ["return","drawdown","probability","pct"] ) and pd.api.types.is_numeric_dtype(o[c]): o[c]=o[c].map(pctstr)
    return o

# ---------- main ----------

def main():
    st.set_page_config(page_title="Multi-Asset Decision Dashboard",layout="wide")
    st.title("Multi-Asset 3–6 Month Decision Support Dashboard")
    st.caption("Optimized build using yfinance + FRED + structured manual inputs.")
    st.sidebar.header("Data settings"); period=st.sidebar.selectbox("Market data history",["5y","10y","15y","max"],index=2); cash=st.sidebar.number_input("Cash / T-bill yield (%)",0.0,10.0,4.5,0.1)
    st.sidebar.header("Ticker settings"); overrides={a:st.sidebar.text_input(f"{a} ticker",cfg["ticker"]) for a,cfg in ASSETS.items()}
    st.sidebar.header("Expected return assumptions"); assum={}
    for a,cfg in ASSETS.items():
        with st.sidebar.expander(a):
            er=st.number_input("Expected annual return (%)",-50.0,100.0,float(cfg["er"]),0.5,key=f"er_{a}"); vp=st.number_input("Valuation/headwind penalty (%)",-20.0,20.0,float(cfg["vp"]),0.5,key=f"vp_{a}"); assum[a]={"er":er,"vp":vp,"cash":cash}
    start=(datetime.now()-timedelta(days=365*16)).strftime("%Y-%m-%d")
    with st.spinner("Loading data..."):
        mkt=load_market(period,overrides); fred=load_fred(start)
    if mkt.empty: st.error("Market data could not be loaded. Check tickers or reboot app."); return
    tabs=st.tabs(["Overview","Manual inputs","S&P 500","Gold","Oil","Bitcoin","FTSE All-World","FTSE Emerging Markets","Macro drivers","Backtest","Diagnostics"])
    with tabs[1]: manual=manual_ui()
    rows=[]
    for i,a in enumerate(ASSETS.keys(),start=2):
        with tabs[i]:
            r=render_asset(a,mkt,fred,assum,manual)
            if r: rows.append(r)
    with tabs[0]:
        st.subheader("Multi-asset overview")
        if rows:
            df=pd.DataFrame(rows)
            for c in ["3m expected return","3m excess vs cash","6m expected return","6m excess vs cash","Mild drawdown probability","Medium drawdown probability","Severe drawdown probability"]: df[c]=df[c].map(pctstr)
            st.dataframe(df,use_container_width=True,hide_index=True)
        st.info("Use this overview for allocation comparison. Manual inputs are included as a weighted overlay.")
    with tabs[8]:
        st.subheader("Macro / cross-asset drivers")
        if not fred.empty:
            for c in ["10Y Treasury Yield","10Y Real Yield","10Y Breakeven Inflation","High Yield Spread","Financial Conditions Index","Trade Weighted Dollar"]:
                if c in fred: st.plotly_chart(px.line(fred[c].dropna(),title=c),use_container_width=True)
        else: st.warning("FRED data unavailable. Check Diagnostics tab.")
        for c in ["US Dollar ETF","WTI Futures","Gold ETF","Bitcoin","VIX"]:
            s=safe(mkt,c)
            if len(s): st.plotly_chart(px.line(s,title=c),use_container_width=True)
    with tabs[9]:
        st.subheader("Backtest / calibration"); st.warning("Backtest excludes manual inputs because historical manual-input logs are not available.")
        a=st.selectbox("Asset to backtest",list(ASSETS.keys())); c1,c2,c3,c4=st.columns(4); h=c1.selectbox("Forward horizon",[63,126],format_func=lambda x:"3 months" if x==63 else "6 months"); sample=c2.number_input("Sample every N trading days",1,21,5,1); low=c3.number_input("Low-risk threshold",0.05,0.50,0.35,0.01); high=c4.number_input("High-risk threshold",0.30,0.90,0.60,0.01)
        if st.button("Run backtest"):
            res,summ=backtest(a,mkt,fred,int(h),int(sample),float(low),float(high)); st.session_state["bt"]=(a,res,summ)
        if "bt" in st.session_state:
            a,res,summ=st.session_state["bt"]
            if res.empty: st.error("Backtest returned no results.")
            else:
                st.dataframe(fmt(summ),use_container_width=True,hide_index=True); x,y=st.columns(2)
                with x:
                    fig=px.box(res,x="Risk bucket",y="Forward return",category_orders={"Risk bucket":["Low","Moderate","High"]},title="Forward return by risk bucket"); fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
                with y:
                    fig=px.box(res,x="Risk bucket",y="Forward max drawdown",category_orders={"Risk bucket":["Low","Moderate","High"]},title="Forward max drawdown by risk bucket"); fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
                raw=res.copy(); raw["Date"]=raw["Date"].dt.strftime("%Y-%m-%d"); st.dataframe(fmt(raw.tail(250)),use_container_width=True,hide_index=True); st.download_button("Download backtest CSV",res.to_csv(index=False),file_name=f"{a.lower().replace(' ','_')}_backtest.csv")
    with tabs[10]:
        st.subheader("Diagnostics"); st.write("Loaded market columns:",list(mkt.columns)); st.write("Latest market date:",str(mkt.dropna(how="all").index[-1].date()))
        if fred.empty:
            d=fred_diag(start); st.write(d); st.error(d.get("error","FRED unavailable"))
        else:
            st.success("FRED data loaded."); st.write("FRED columns:",list(fred.columns)); st.write("Latest FRED date:",str(fred.dropna(how="all").index[-1].date()))
    st.caption("Decision-support tool only; not financial advice. Use it to structure judgement around risk, fragility and holding attractiveness.")

if __name__=="__main__": main()
