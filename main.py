#!/usr/bin/env python3
import os, time, math, random, logging
from collections import deque
from dataclasses import dataclass
import requests

# ===================== CONFIG =====================

EXCHANGE = os.getenv("EXCHANGE","coinbase").lower()
API_BASE = os.getenv("API_BASE","https://crypto-ai-api-1-7cte.onrender.com")
VAULT_PIN = os.getenv("VAULT_PIN","4567")

CYCLE_SECONDS = int(os.getenv("CYCLE_SECONDS",15))
HISTORY_LEN = int(os.getenv("HISTORY_LEN",30))
MIN_CONF = float(os.getenv("MIN_CONF",0.20))

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE",0.015))
MAX_RISK_MULT = float(os.getenv("MAX_RISK_MULT",1.5))

STARTING_EQUITY = float(os.getenv("STARTING_EQUITY",1000))
MAX_POS_ABS = float(os.getenv("MAX_POS_ABS",100))
MAX_POS_PCT = float(os.getenv("MAX_POS_PCT",0.10))
DAILY_LOSS_CAP = float(os.getenv("DAILY_LOSS_CAP",0.03))

TAKER_FEE = 0.004
SLIPPAGE = 0.001

UNIVERSE = os.getenv("UNIVERSE","BTC-USD,ETH-USD,SOL-USD,ADA-USD,XRP-USD,DOGE-USD").split(",")

LIVE_MODE = os.getenv("LIVE_MODE","0") == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("BOT")

session = requests.Session()

# ===================== EXCHANGE ADAPTERS =====================

class Coinbase:
    base = "https://api.exchange.coinbase.com"

    def products(self):
        r = session.get(self.base+"/products",timeout=10)
        return [p["id"] for p in r.json()]

    def ticker(self,s):
        r = session.get(self.base+f"/products/{s}/ticker",timeout=10)
        return float(r.json()["price"])

class KuCoin:
    base = "https://api.kucoin.com"

    def products(self):
        r = session.get(self.base+"/api/v2/symbols",timeout=10)
        return [p["symbol"] for p in r.json()["data"]]

    def ticker(self,s):
        s = s.replace("-","")
        r = session.get(self.base+"/api/v1/market/orderbook/level1",params={"symbol":s},timeout=10)
        return float(r.json()["data"]["price"])

ex = Coinbase() if EXCHANGE=="coinbase" else KuCoin()

# ===================== BRAIN V3 =====================

class BrainV3:
    def evaluate(self,f):
        mom = f["momentum"]
        vol = f["vol"]
        cons = f["consistency"]
        score = (mom*1.2)+(cons*0.8)-(vol*1.5)
        conf = 1/(1+math.exp(-score))
        allow = conf >= MIN_CONF and vol < 2
        return {"allow":allow,"confidence":conf,"reason":"ok"}

brain = BrainV3()

# ===================== PAPER BROKER =====================

@dataclass
class Position:
    market:str
    entry:float
    qty:float
    cycle:int

class Broker:
    def __init__(self):
        self.equity = STARTING_EQUITY
        self.day_start = STARTING_EQUITY
        self.pos = None

    def daily_loss_hit(self):
        return (self.day_start-self.equity)/self.day_start > DAILY_LOSS_CAP

    def size(self,rm):
        raw = self.equity*RISK_PER_TRADE*rm
        return min(raw,self.equity*MAX_POS_PCT,MAX_POS_ABS)

    def open(self,m,p,rm,c):
        v = self.size(rm)
        qty = (v*(1-TAKER_FEE))/(p*(1+SLIPPAGE))
        self.pos = Position(m,p,qty,c)
        return v,qty

    def close(self,p):
        exitp = p*(1-SLIPPAGE)
        pnl = (exitp-self.pos.entry)*self.pos.qty
        self.equity += pnl
        self.pos=None
        return pnl

broker = Broker()

# ===================== HELPERS =====================

def momentum(pr):
    return ((pr[-1]-pr[0])/pr[0])*100

def stdev(xs):
    if len(xs)<2: return 0
    m=sum(xs)/len(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

# ===================== INIT =====================

log.info("Booting bot on %s",EXCHANGE)
products = ex.products()
UNIVERSE = [u for u in UNIVERSE if u.replace("-","") in "".join(products)]
log.info("Universe: %s",UNIVERSE)

prices = {m:deque(maxlen=HISTORY_LEN) for m in UNIVERSE}

# ===================== MAIN LOOP =====================

cycle=0
while True:
    cycle+=1

    if broker.daily_loss_hit():
        log.warning("DAILY LOSS CAP HIT")
        time.sleep(60)
        continue

    best=None; best_score=-999; best_price=None; best_feat=None

    for m in UNIVERSE:
        try:
            p = ex.ticker(m)
        except:
            continue
        prices[m].append(p)
        if len(prices[m])<5: continue

        mom = momentum(prices[m])
        rets=[(prices[m][i]-prices[m][i-1])/prices[m][i-1] for i in range(1,len(prices[m]))]
        vol = stdev(rets)
        cons = sum(1 for r in rets if r>0)/len(rets)
        score = mom-(vol*0.4)

        if score>best_score:
            best_score=score
            best=m
            best_price=p
            best_feat={"momentum":mom,"vol":vol,"consistency":cons}

    if not best:
        log.info("SKIP no data")
        time.sleep(CYCLE_SECONDS)
        continue

    gate = brain.evaluate(best_feat)
    conf = gate["confidence"]
    rm = 1+(conf*(MAX_RISK_MULT-1))

    if broker.pos:
        if cycle-broker.pos.cycle>=1 or conf<MIN_CONF:
            pnl = broker.close(best_price)
            log.info("CLOSE %s pnl=%.2f equity=%.2f",best,pnl,broker.equity)
    else:
        if conf>=MIN_CONF and gate["allow"]:
            v,qty = broker.open(best,best_price,rm,cycle)
            log.info("OPEN %s conf=%.2f size=%.2f equity=%.2f",best,conf,v,broker.equity)
        else:
            log.info("SKIP %s conf=%.2f",best,conf)

    time.sleep(CYCLE_SECONDS)
