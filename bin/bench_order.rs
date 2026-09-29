use eth_tx_order::{AccountNonce, ConflictPolicy, Orderer, Tip256, TxMeta};
use std::hint::black_box;
use std::time::{Duration, Instant};

const DEFAULT_TRANSACTIONS: usize = 100_000;
const DEFAULT_SENDERS: usize = 80_000;
const WARMUP_RUNS: usize = 10;
const MEASURE_RUNS: usize = 100;
const DEFAULT_ROTATING_BATCHES: usize = 8;

fn main() {
    let count = argument(1).unwrap_or(DEFAULT_TRANSACTIONS);
    let senders = argument(2).unwrap_or(DEFAULT_SENDERS).clamp(1, count.max(1));
    let runs = argument(3).unwrap_or(MEASURE_RUNS).max(1);
    let dense = std::env::args().nth(4).as_deref() != Some("checked");
    let rotating_batches = argument(5).unwrap_or(DEFAULT_ROTATING_BATCHES).max(1);

    let batches: Vec<Batch> = (0..rotating_batches)
        .map(|batch| {
            make_batch(
                count,
                senders,
                0x243f_6a88_85a3_08d3 ^ mix64(batch as u64),
            )
        })
        .collect();
    let mut orderer = if dense {
        Orderer::with_dense_capacity(count)
    } else {
        Orderer::with_capacity(count)
    }
    .expect("valid capacity");
    let mut output = Vec::with_capacity(count);

    for run in 0..WARMUP_RUNS {
        let batch = &batches[run % batches.len()];
        let ordered = run_batch(&mut orderer, batch, &mut output, dense);
        black_box(ordered);
    }
    assert_eq!(output.len(), count, "synthetic batch must be fully ordered");

    let mut samples = Vec::with_capacity(runs);
    for run in 0..runs {
        let batch = &batches[run % batches.len()];
        let start = Instant::now();
        let ordered = run_batch(&mut orderer, batch, &mut output, dense);
        let elapsed = start.elapsed();
        black_box(ordered);
        black_box(&output);
        samples.push(elapsed);
    }

    samples.sort_unstable();
    let total: Duration = samples.iter().copied().sum();
    let mean = total / samples.len() as u32;
    let p50 = percentile(&samples, 50);
    let p95 = percentile(&samples, 95);
    let p99 = percentile(&samples, 99);
    let throughput = count as f64 / mean.as_secs_f64();

    println!("transactions: {count}");
    println!("senders:      {senders}");
    println!("runs:         {runs}");
    println!("mode:         {}", if dense { "trusted-dense" } else { "checked" });
    println!("batch rotation: {rotating_batches}");
    println!("mean:         {:.3} ms", mean.as_secs_f64() * 1_000.0);
    println!("p50:          {:.3} ms", p50.as_secs_f64() * 1_000.0);
    println!("p95:          {:.3} ms", p95.as_secs_f64() * 1_000.0);
    println!("p99:          {:.3} ms", p99.as_secs_f64() * 1_000.0);
    println!("throughput:   {:.0} tx/s", throughput);
}

fn run_batch(
    orderer: &mut Orderer,
    batch: &Batch,
    output: &mut Vec<u32>,
    dense: bool,
) -> usize {
    if dense {
        orderer
            .order_trusted_dense(
                black_box(&batch.transactions),
                black_box(&batch.account_ids),
                black_box(&batch.base_nonces),
                black_box(output),
                ConflictPolicy::KeepHigherPriority,
            )
            .expect("valid dense batch")
    } else {
        orderer
            .order_with_state(
                black_box(&batch.transactions),
                black_box(&batch.state),
                black_box(output),
                ConflictPolicy::KeepHigherPriority,
            )
            .expect("valid checked batch")
            .ordered_transactions
    }
}

fn argument(position: usize) -> Option<usize> {
    std::env::args().nth(position)?.parse().ok()
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let index = (samples.len() - 1) * percentile / 100;
    samples[index]
}

struct Batch {
    transactions: Vec<TxMeta>,
    state: Vec<AccountNonce>,
    account_ids: Vec<u32>,
    base_nonces: Vec<u64>,
}

fn make_batch(count: usize, sender_count: usize, seed: u64) -> Batch {
    let mut random = SplitMix64(seed);
    let mut state = Vec::with_capacity(sender_count);
    for sender_id in 0..sender_count {
        state.push(AccountNonce {
            sender: address(sender_id as u64),
            next_nonce: 0,
        });
    }

    let mut next_nonce = vec![0u64; sender_count];
    let mut transactions = Vec::with_capacity(count);
    let mut account_ids = Vec::with_capacity(count);
    for index in 0..count {
        let sender_id = index % sender_count;
        let nonce = next_nonce[sender_id];
        next_nonce[sender_id] += 1;
        transactions.push(TxMeta {
            effective_tip: Tip256::from_u64(random.next() % 100_000),
            hash: transaction_hash(index as u64, random.next()),
            sender: state[sender_id].sender,
            nonce,
        });
        account_ids.push(sender_id as u32);
    }

    // Do not benchmark an input that is accidentally favorable to any one
    // account or priority pattern.
    for index in (1..transactions.len()).rev() {
        let target = (random.next() as usize) % (index + 1);
        transactions.swap(index, target);
        account_ids.swap(index, target);
    }
    Batch {
        transactions,
        state,
        account_ids,
        base_nonces: vec![0; sender_count],
    }
}

fn address(id: u64) -> [u8; 20] {
    let mut address = [0u8; 20];
    let mixed = mix64(id ^ 0xa409_3822_299f_31d0);
    address[0..8].copy_from_slice(&mixed.to_le_bytes());
    address[8..16].copy_from_slice(&mix64(mixed).to_le_bytes());
    address[16..20].copy_from_slice(&(mix64(mixed ^ id) as u32).to_le_bytes());
    address
}

fn transaction_hash(id: u64, random: u64) -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0..8].copy_from_slice(&mix64(id).to_le_bytes());
    hash[8..16].copy_from_slice(&mix64(random).to_le_bytes());
    hash[16..24].copy_from_slice(&mix64(id ^ random).to_le_bytes());
    hash[24..32].copy_from_slice(&id.to_be_bytes());
    hash
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
