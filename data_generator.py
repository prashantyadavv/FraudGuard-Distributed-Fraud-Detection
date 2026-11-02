"""Generate PaySim-shaped data, including explainable fraud patterns."""
import argparse, csv, random
from pathlib import Path
from src.config import DATA_DIR, PAYSIM_COLUMNS, SEED, ensure_directories

def transaction(step, kind, amount, origin, dest, old_org, old_dest, fraud=0):
    """Return one schema-compatible row; balances deliberately follow the transfer."""
    new_org = max(0.0, old_org - amount)
    new_dest = old_dest + amount
    return [step, kind, round(amount,2), origin, round(old_org,2), round(new_org,2), dest, round(old_dest,2), round(new_dest,2), fraud, int(fraud and amount > 200000)]

def generate(rows: int, output: Path, fraud_rate: float = .005, seed: int = SEED) -> Path:
    """Create a reproducible dataset with draining chains, rings, and mule receipts.

    The normal transactions dominate, so the class imbalance resembles fraud work.
    """
    random.seed(seed); ensure_directories(); output.parent.mkdir(parents=True, exist_ok=True)
    accounts = [f"C{i:08d}" for i in range(max(1000, rows // 8))]
    merchants = [f"M{i:06d}" for i in range(100)]
    target_fraud = max(1, int(rows * fraud_rate)); normal_count = rows - target_fraud
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(PAYSIM_COLUMNS)
        for i in range(normal_count):
            src, dst = random.choice(accounts), random.choice(accounts + merchants)
            amount, balance = random.expovariate(1/25000)+10, random.uniform(1000, 300000)
            writer.writerow(transaction(i % 744, random.choices(["PAYMENT","CASH_OUT","TRANSFER","DEBIT"],[.55,.25,.17,.03])[0], amount, src, dst, balance, random.uniform(0, 200000)))
        # TRANSFER -> CASH_OUT: a classic PaySim-style draining chain.
        for i in range(target_fraud // 2):
            victim, mule = random.choice(accounts), f"C_MULE_{i%20:04d}"; amount = random.uniform(50000, 300000); step = i % 744
            writer.writerow(transaction(step, "TRANSFER", amount, victim, mule, amount, random.uniform(0,5000), 1))
        # A repeating directed ring makes motif queries meaningful.
        ring = [f"C_RING_{x:02d}" for x in range(8)]
        for i in range(target_fraud - target_fraud // 2):
            src, dst = ring[i % len(ring)], ring[(i+1) % len(ring)]; amount = random.uniform(20000, 100000)
            writer.writerow(transaction((i+50)%744, "TRANSFER", amount, src, dst, amount, 0, 1))
    return output

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--rows", type=int, default=100000, choices=[100000,500000,1000000,3000000]); parser.add_argument("--fraud-rate", type=float, default=.005); parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic_paysim_100000.csv")
    args = parser.parse_args(); print("Created", generate(args.rows, args.output, args.fraud_rate))
if __name__ == "__main__": main()
