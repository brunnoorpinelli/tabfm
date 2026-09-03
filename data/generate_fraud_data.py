"""
Gerador de dados sintéticos de transações de cartão para demo de detecção de fraude
com TabFM (AI.PREDICT) no BigQuery.

Objetivo: dataset com relevância estatística (~120k transações, ~2,5% de fraude),
sinal realista (não trivial) e um "padrão novo" de fraude que surge a partir de junho,
para demonstrar adaptação sem re-treino.

Uso:
    python generate_fraud_data.py --rows 120000 --seed 42 --out transactions.csv
    from generate_fraud_data import generate_transactions
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "supermercado", "restaurante", "combustivel", "farmacia", "vestuario",
    "servicos_publicos", "eletronicos", "viagem", "games_digitais", "cripto",
]
# Probabilidade de cada categoria em transações legítimas vs. fraudulentas
LEGIT_MCC_P = np.array([0.22, 0.18, 0.12, 0.08, 0.10, 0.08, 0.08, 0.06, 0.05, 0.03])
FRAUD_MCC_P = np.array([0.04, 0.04, 0.03, 0.02, 0.06, 0.02, 0.28, 0.16, 0.18, 0.17])

CHANNELS = ["pos_chip", "pos_tarja", "online", "aproximacao", "caixa_eletronico"]
LEGIT_CH_P = np.array([0.30, 0.05, 0.35, 0.25, 0.05])
FRAUD_CH_P = np.array([0.05, 0.22, 0.58, 0.10, 0.05])

START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-08-01")  # exclusivo


def _hours(rng: np.random.Generator, n: int, fraud: bool) -> np.ndarray:
    if fraud:
        # fraude concentrada de madrugada
        mix = rng.random(n) < 0.55
        night = rng.integers(0, 6, n)
        day = rng.integers(6, 24, n)
        return np.where(mix, night, day)
    # legítimo: pico comercial e noite
    return np.clip(rng.normal(14, 5, n).round().astype(int), 0, 23)


def generate_transactions(n_rows: int = 120_000, fraud_rate: float = 0.025,
                          seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # ---------- população de clientes ----------
    n_customers = 20_000
    cust_age = np.clip(rng.normal(42, 14, n_customers).round(), 18, 85).astype(int)
    cust_acct_age = np.clip(rng.exponential(900, n_customers).round(), 5, 6000).astype(int)
    cust_avg_amt = np.clip(rng.lognormal(4.3, 0.6, n_customers), 20, 3000)
    cust_limit = np.clip(cust_avg_amt * rng.uniform(8, 40, n_customers), 500, 60000).round(-2)
    cust_prior_cb = rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], n_customers)

    def build(n: int, fraud: bool) -> pd.DataFrame:
        # fraudadores atacam preferencialmente contas novas e com histórico ruim
        if fraud:
            w = 1.0 / (cust_acct_age + 60.0) * (1 + cust_prior_cb)
            w /= w.sum()
            cid = rng.choice(n_customers, n, p=w)
        else:
            cid = rng.integers(0, n_customers, n)

        ts = START + pd.to_timedelta(rng.uniform(0, (END - START).total_seconds(), n), unit="s")
        hour = _hours(rng, n, fraud)
        ts = ts.normalize() + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")

        mcc = rng.choice(MERCHANT_CATEGORIES, n, p=FRAUD_MCC_P if fraud else LEGIT_MCC_P)
        ch = rng.choice(CHANNELS, n, p=FRAUD_CH_P if fraud else LEGIT_CH_P)
        card_present = np.isin(ch, ["pos_chip", "pos_tarja", "aproximacao", "caixa_eletronico"])

        avg30 = cust_avg_amt[cid] * rng.lognormal(0, 0.15, n)
        if fraud:
            # bimodal: "teste de cartão" (micro) ou "esvaziar limite" (alto)
            micro = rng.random(n) < 0.25
            amt = np.where(micro,
                           rng.uniform(1, 15, n),
                           avg30 * rng.lognormal(1.4, 0.7, n))
        else:
            amt = avg30 * rng.lognormal(0, 0.55, n)
        amt = np.clip(amt, 1, 50000).round(2)

        if fraud:
            intl = rng.random(n) < 0.35
            dist = np.where(intl, rng.uniform(1500, 12000, n), rng.exponential(120, n))
            new_dev = (rng.random(n) < 0.70) & ~card_present
            txn24 = rng.poisson(5.5, n) + 1
            merch24 = np.minimum(txn24, rng.poisson(3.5, n) + 1)
            failed = rng.poisson(1.6, n)
            util = np.clip(rng.beta(4, 3, n), 0, 1)
        else:
            intl = rng.random(n) < 0.03
            dist = np.where(intl, rng.uniform(1500, 12000, n), rng.exponential(12, n))
            new_dev = (rng.random(n) < 0.08) & ~card_present
            txn24 = rng.poisson(1.6, n) + 1
            merch24 = np.minimum(txn24, rng.poisson(1.2, n) + 1)
            failed = rng.poisson(0.12, n)
            util = np.clip(rng.beta(2, 5, n), 0, 1)

        df = pd.DataFrame({
            "customer_id": cid,
            "transaction_ts": ts,
            "amount": amt,
            "merchant_category": mcc,
            "channel": ch,
            "card_present": card_present,
            "is_international": intl,
            "distance_from_home_km": dist.round(1),
            "new_device": new_dev,
            "txn_count_24h": txn24,
            "distinct_merchants_24h": merch24,
            "failed_auth_24h": failed,
            "customer_age": cust_age[cid],
            "account_age_days": cust_acct_age[cid],
            "avg_amount_30d": avg30.round(2),
            "credit_limit_utilization": util.round(3),
            "prior_chargebacks": cust_prior_cb[cid],
            "is_fraud": fraud,
        })
        return df

    legit = build(n_legit, fraud=False)
    fraud = build(n_fraud, fraud=True)

    # ---------- "fraude difícil": 30% das fraudes camufladas como legítimas ----------
    n_hard = int(len(fraud) * 0.30)
    hard_idx = rng.choice(len(fraud), n_hard, replace=False)
    camo = build(n_hard, fraud=False)
    for col in ["amount", "merchant_category", "channel", "card_present", "distance_from_home_km",
                "is_international", "txn_count_24h", "distinct_merchants_24h", "failed_auth_24h",
                "credit_limit_utilization"]:
        fraud.loc[hard_idx, col] = camo[col].values
    # mantém sinal fraco: dispositivo novo e horário
    fraud.loc[hard_idx, "new_device"] = (rng.random(n_hard) < 0.45) & ~fraud.loc[hard_idx, "card_present"].values

    # ---------- ruído em legítimas: 3% parecem suspeitas (viagens, compras grandes) ----------
    n_noisy = int(len(legit) * 0.03)
    noisy_idx = rng.choice(len(legit), n_noisy, replace=False)
    legit.loc[noisy_idx, "is_international"] = True
    legit.loc[noisy_idx, "distance_from_home_km"] = rng.uniform(1500, 9000, n_noisy).round(1)
    legit.loc[noisy_idx, "amount"] = (legit.loc[noisy_idx, "avg_amount_30d"] * rng.lognormal(1.0, 0.5, n_noisy)).round(2)
    legit.loc[noisy_idx, "merchant_category"] = rng.choice(["viagem", "eletronicos", "restaurante"], n_noisy)

    # ---------- padrão NOVO de fraude a partir de 2026-06-01 ----------
    # "Ataque de aproximação em games digitais": muitos micro-pagamentos por aproximação,
    # cartão presente, perto de casa. Não existe antes de junho -> demonstra adaptação sem re-treino.
    n_new = int(n_rows * 0.006)
    new = build(n_new, fraud=False)  # base legítima, depois sobrescreve o padrão
    new["transaction_ts"] = pd.Timestamp("2026-06-01") + pd.to_timedelta(
        rng.uniform(0, (END - pd.Timestamp("2026-06-01")).total_seconds(), n_new), unit="s")
    new["merchant_category"] = "games_digitais"
    new["channel"] = "aproximacao"
    new["card_present"] = True
    new["amount"] = rng.uniform(19, 49, n_new).round(2)
    new["txn_count_24h"] = rng.poisson(9, n_new) + 4
    new["distinct_merchants_24h"] = 1
    new["distance_from_home_km"] = rng.exponential(3, n_new).round(1)
    new["is_international"] = False
    new["new_device"] = False
    new["is_fraud"] = True

    df = pd.concat([legit, fraud, new], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{seed:02d}{i:08d}" for i in range(len(df))])

    # features derivadas (ficam ≤ 20 features no total)
    df["amount_to_avg_ratio"] = (df["amount"] / df["avg_amount_30d"]).round(3)
    df["hour_of_day"] = df["transaction_ts"].dt.hour.astype(int)
    df["day_of_week"] = df["transaction_ts"].dt.dayofweek.astype(int)
    df["transaction_ts"] = df["transaction_ts"].dt.floor("s")

    cols = ["transaction_id", "customer_id", "transaction_ts", "amount", "merchant_category", "channel",
            "card_present", "is_international", "distance_from_home_km", "new_device", "txn_count_24h",
            "distinct_merchants_24h", "failed_auth_24h", "customer_age", "account_age_days",
            "avg_amount_30d", "amount_to_avg_ratio", "credit_limit_utilization", "prior_chargebacks",
            "hour_of_day", "day_of_week", "is_fraud"]
    return df[cols]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--fraud-rate", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="transactions.csv")
    a = ap.parse_args()
    df = generate_transactions(a.rows, a.fraud_rate, a.seed)
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} linhas | fraude={df.is_fraud.mean():.2%} | salvo em {a.out}")
