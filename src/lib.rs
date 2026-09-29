//! Fast deterministic ordering for prevalidated Ethereum transactions.
//!
//! Ethereum nonce dependencies are a disjoint union of per-sender chains. This
//! crate exploits that structure rather than constructing a general DAG.
//! Transactions are ordered by effective tip descending and transaction hash
//! ascending among the transactions that are nonce-ready at each step.
//!
//! The hot path uses reusable open-addressed index tables and adapts scheduling
//! to the number of live account chains: direct scanning for very small sets, a
//! fixed-leaf tournament for moderate sets, and compact priority ranking plus a
//! hierarchical bitset for large sets. Once an [`Orderer`] has sufficient
//! capacity, ordering performs no allocation if the caller also reuses the
//! output vector.

use core::cmp::Ordering;
use core::fmt;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

/// An Ethereum address.
pub type Address = [u8; 20];

/// A transaction hash. Hash ties are broken in ascending byte order.
pub type TxHash = [u8; 32];

const NONE: u32 = u32::MAX;
const LINEAR_READY_LIMIT: usize = 1;
const TOURNAMENT_READY_LIMIT: usize = 1;

/// An unsigned 256-bit effective tip, stored as little-endian 64-bit limbs.
///
/// `limbs[0]` is the least significant limb and `limbs[3]` is the most
/// significant limb. This avoids imposing a particular Ethereum primitive
/// types dependency on the hot path.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tip256 {
    pub limbs: [u64; 4],
}

impl Tip256 {
    pub const ZERO: Self = Self { limbs: [0; 4] };

    #[inline(always)]
    pub const fn from_u64(value: u64) -> Self {
        Self {
            limbs: [value, 0, 0, 0],
        }
    }

    #[inline(always)]
    pub const fn from_u128(value: u128) -> Self {
        Self {
            limbs: [value as u64, (value >> 64) as u64, 0, 0],
        }
    }
}

impl Ord for Tip256 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs[3]
            .cmp(&other.limbs[3])
            .then_with(|| self.limbs[2].cmp(&other.limbs[2]))
            .then_with(|| self.limbs[1].cmp(&other.limbs[1]))
            .then_with(|| self.limbs[0].cmp(&other.limbs[0]))
    }
}

impl PartialOrd for Tip256 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Metadata required by the ordering stage.
///
/// Hashing, signature recovery, transaction decoding, fee validation and
/// computation of `effective_tip` should happen upstream. Returning indices
/// into this slice avoids copying complete transactions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TxMeta {
    /// `min(max_priority_fee_per_gas, max_fee_per_gas - block_base_fee)` for a
    /// valid EIP-1559 transaction, or the corresponding effective tip for the
    /// transaction type in use.
    pub effective_tip: Tip256,
    pub hash: TxHash,
    pub sender: Address,
    pub nonce: u64,
}

/// The next executable nonce for an account in the parent state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccountNonce {
    pub sender: Address,
    pub next_nonce: u64,
}

/// Policy for distinct transactions occupying the same `(sender, nonce)` slot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConflictPolicy {
    /// Keep the transaction with the higher effective tip, breaking a tie by
    /// the lower transaction hash. This is the fastest useful block-building
    /// policy and is deterministic.
    #[default]
    KeepHigherPriority,
    /// Return an error on the first distinct same-nonce conflict.
    RejectBatch,
}

/// Counters from one ordering run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrderStats {
    pub input_transactions: usize,
    pub unique_hashes: usize,
    pub exact_duplicates: usize,
    pub stale_transactions: usize,
    pub nonce_conflicts: usize,
    pub selected_nonce_slots: usize,
    pub blocked_by_nonce_gap: usize,
    pub ordered_transactions: usize,
}

/// An input error. Ordinary duplicates, stale transactions and nonce gaps are
/// reported in [`OrderStats`] rather than treated as errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrderError {
    TooManyTransactions { count: usize },
    DenseAccountLengthMismatch {
        transactions: usize,
        account_ids: usize,
    },
    DenseAccountOutOfRange {
        transaction_index: usize,
        account_id: u32,
        account_count: usize,
    },
    MissingAccountState { sender: Address },
    ConflictingAccountState {
        sender: Address,
        first_nonce: u64,
        second_nonce: u64,
    },
    InconsistentDuplicateHash { hash: TxHash },
    ConflictingNonce { sender: Address, nonce: u64 },
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyTransactions { count } => {
                write!(f, "batch has {count} transactions; u32 indexing is required")
            }
            Self::DenseAccountLengthMismatch {
                transactions,
                account_ids,
            } => write!(
                f,
                "dense account-id length mismatch: {transactions} transactions and {account_ids} ids"
            ),
            Self::DenseAccountOutOfRange {
                transaction_index,
                account_id,
                account_count,
            } => write!(
                f,
                "transaction {transaction_index} has dense account id {account_id}, but only {account_count} base nonces were supplied"
            ),
            Self::MissingAccountState { sender } => {
                write!(f, "missing parent-state nonce for 0x")?;
                fmt_hex(f, sender)
            }
            Self::ConflictingAccountState {
                sender,
                first_nonce,
                second_nonce,
            } => {
                write!(f, "conflicting parent-state nonces for 0x")?;
                fmt_hex(f, sender)?;
                write!(f, ": {first_nonce} and {second_nonce}")
            }
            Self::InconsistentDuplicateHash { hash } => {
                write!(f, "the same transaction hash has inconsistent metadata: 0x")?;
                fmt_hex(f, hash)
            }
            Self::ConflictingNonce { sender, nonce } => {
                write!(f, "distinct transactions conflict at 0x")?;
                fmt_hex(f, sender)?;
                write!(f, " nonce {nonce}")
            }
        }
    }
}

impl std::error::Error for OrderError {}

fn fmt_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

/// Reusable state for the ordering algorithm.
///
/// Construct one orderer per worker and retain it across batches. An orderer is
/// deliberately not synchronized; separate workers should not contend on the
/// same workspace.
pub struct Orderer {
    hashes: IndexTable,
    slots: IndexTable,
    account_index: IndexTable,
    accounts: Vec<AccountNonce>,
    base_nonces: Vec<u64>,
    reachable: Vec<u32>,
    heads: Vec<u32>,
    successor: Vec<u32>,
    rank: Vec<u32>,
    account_of: Vec<u32>,
    rank64: Vec<Rank64>,
    rank256: Vec<Rank256>,
    tournament: Vec<u32>,
    ready: HierarchicalBitSet,
}

impl Default for Orderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Orderer {
    #[inline]
    pub fn new() -> Self {
        let seed = HashSeed::random();
        Self {
            hashes: IndexTable::new(seed.derive(0x243f_6a88_85a3_08d3)),
            slots: IndexTable::new(seed.derive(0x1319_8a2e_0370_7344)),
            account_index: IndexTable::new(seed.derive(0xa409_3822_299f_31d0)),
            accounts: Vec::new(),
            base_nonces: Vec::new(),
            reachable: Vec::new(),
            heads: Vec::new(),
            successor: Vec::new(),
            rank: Vec::new(),
            account_of: Vec::new(),
            rank64: Vec::new(),
            rank256: Vec::new(),
            tournament: Vec::new(),
            ready: HierarchicalBitSet::new(),
        }
    }

    /// Preallocate for a maximum batch size. This should be called outside the
    /// latency-critical path.
    pub fn with_capacity(max_transactions: usize) -> Result<Self, OrderError> {
        let mut orderer = Self::new();
        orderer.reserve(max_transactions)?;
        Ok(orderer)
    }

    /// Preallocate only the structures touched by [`Self::order_trusted_dense`].
    /// Checked APIs remain usable and will allocate their additional tables on
    /// first use.
    pub fn with_dense_capacity(max_transactions: usize) -> Result<Self, OrderError> {
        let mut orderer = Self::new();
        orderer.reserve_dense(max_transactions)?;
        Ok(orderer)
    }

    /// Ensure that the workspace can process a batch of this size without
    /// growing its principal buffers.
    pub fn reserve(&mut self, max_transactions: usize) -> Result<(), OrderError> {
        check_batch_size(max_transactions)?;
        self.hashes.ensure_capacity(max_transactions)?;
        self.account_index.ensure_capacity(max_transactions)?;
        reserve_to(&mut self.accounts, max_transactions);
        self.reserve_dense(max_transactions)
    }

    /// Reserve the trusted dense-account fast path without allocating checked
    /// metadata tables.
    pub fn reserve_dense(&mut self, max_transactions: usize) -> Result<(), OrderError> {
        check_batch_size(max_transactions)?;
        self.slots.ensure_capacity(max_transactions)?;
        reserve_to(&mut self.base_nonces, max_transactions);
        reserve_to(&mut self.reachable, max_transactions);
        reserve_to(&mut self.heads, max_transactions);
        reserve_to(&mut self.successor, max_transactions);
        reserve_to(&mut self.rank, max_transactions);
        reserve_to(&mut self.account_of, max_transactions);
        reserve_to(&mut self.rank64, max_transactions);
        reserve_to(&mut self.rank256, max_transactions);
        reserve_to(
            &mut self.tournament,
            TOURNAMENT_READY_LIMIT.next_power_of_two() * 2,
        );
        self.ready.ensure_universe(max_transactions);
        Ok(())
    }

    /// Order transactions using authoritative parent-state account nonces.
    ///
    /// `account_nonces` need contain only the distinct senders present in the
    /// batch. Extra entries are harmless. The returned indices address `txs`.
    pub fn order_with_state(
        &mut self,
        txs: &[TxMeta],
        account_nonces: &[AccountNonce],
        output: &mut Vec<u32>,
        conflict_policy: ConflictPolicy,
    ) -> Result<OrderStats, OrderError> {
        output.clear();
        check_batch_size(txs.len())?;
        self.build_accounts_from_state(account_nonces)?;
        self.order_prepared_accounts::<false>(txs, &[], output, conflict_policy)
    }

    /// Order a self-contained batch by treating each sender's minimum observed
    /// nonce as immediately executable.
    ///
    /// This is useful for benchmarking and for already state-trimmed batches.
    /// Consensus code should normally use [`Self::order_with_state`].
    pub fn order_assuming_min_nonce(
        &mut self,
        txs: &[TxMeta],
        output: &mut Vec<u32>,
        conflict_policy: ConflictPolicy,
    ) -> Result<OrderStats, OrderError> {
        output.clear();
        check_batch_size(txs.len())?;
        self.build_accounts_from_minimum(txs)?;
        self.order_prepared_accounts::<false>(txs, &[], output, conflict_policy)
    }

    /// Fast path for metadata that has already been canonicalized, validated
    /// and assigned dense account identifiers upstream.
    ///
    /// `account_ids[i]` identifies `txs[i]`, and `base_nonces[id]` is that
    /// account's authoritative parent-state nonce. Account ids must identify
    /// senders exactly: `account_ids[i] == account_ids[j]` if and only if the
    /// two transactions have the same sender. Every sender must therefore use
    /// exactly one id. The caller must also guarantee that equal transaction
    /// hashes have identical metadata. Under those invariants, exact duplicates
    /// necessarily collide in the same nonce slot, so this path omits both
    /// address interning and the global hash table. Violating these trusted
    /// semantic preconditions can produce an invalid nonce order.
    ///
    /// The return value is the number of ordered transactions. Use
    /// [`Self::order_with_state`] when metadata is not already trusted or when
    /// detailed duplicate statistics are required.
    pub fn order_trusted_dense(
        &mut self,
        txs: &[TxMeta],
        account_ids: &[u32],
        base_nonces: &[u64],
        output: &mut Vec<u32>,
        conflict_policy: ConflictPolicy,
    ) -> Result<usize, OrderError> {
        output.clear();
        check_batch_size(txs.len())?;
        check_batch_size(base_nonces.len())?;
        if account_ids.len() != txs.len() {
            return Err(OrderError::DenseAccountLengthMismatch {
                transactions: txs.len(),
                account_ids: account_ids.len(),
            });
        }
        self.base_nonces.clear();
        self.base_nonces.extend_from_slice(base_nonces);
        let stats = self.order_prepared_accounts::<true>(
            txs,
            account_ids,
            output,
            conflict_policy,
        )?;
        Ok(stats.ordered_transactions)
    }

    fn build_accounts_from_state(
        &mut self,
        account_nonces: &[AccountNonce],
    ) -> Result<(), OrderError> {
        check_batch_size(account_nonces.len())?;
        self.accounts.clear();
        self.base_nonces.clear();
        self.account_index.start_batch(account_nonces.len())?;

        for account in account_nonces.iter().copied() {
            if let Some(existing) = self
                .account_index
                .find_account(&self.accounts, &account.sender)
            {
                let first = self.accounts[existing as usize].next_nonce;
                if first != account.next_nonce {
                    return Err(OrderError::ConflictingAccountState {
                        sender: account.sender,
                        first_nonce: first,
                        second_nonce: account.next_nonce,
                    });
                }
                continue;
            }

            let index = self.accounts.len() as u32;
            self.accounts.push(account);
            self.base_nonces.push(account.next_nonce);
            self.account_index
                .insert_account(&self.accounts, index);
        }
        Ok(())
    }

    fn build_accounts_from_minimum(&mut self, txs: &[TxMeta]) -> Result<(), OrderError> {
        self.accounts.clear();
        self.base_nonces.clear();
        self.account_index.start_batch(txs.len())?;

        for tx in txs {
            if let Some(existing) = self.account_index.find_account(&self.accounts, &tx.sender) {
                let account = &mut self.accounts[existing as usize];
                account.next_nonce = account.next_nonce.min(tx.nonce);
                self.base_nonces[existing as usize] = account.next_nonce;
                continue;
            }

            let index = self.accounts.len() as u32;
            self.accounts.push(AccountNonce {
                sender: tx.sender,
                next_nonce: tx.nonce,
            });
            self.base_nonces.push(tx.nonce);
            self.account_index
                .insert_account(&self.accounts, index);
        }
        Ok(())
    }

    fn order_prepared_accounts<const TRUSTED_DENSE: bool>(
        &mut self,
        txs: &[TxMeta],
        dense_account_ids: &[u32],
        output: &mut Vec<u32>,
        conflict_policy: ConflictPolicy,
    ) -> Result<OrderStats, OrderError> {
        if !TRUSTED_DENSE {
            self.hashes.start_batch(txs.len())?;
        }
        self.slots.start_batch(txs.len())?;
        self.reachable.clear();
        self.heads.clear();
        self.successor.resize(txs.len(), NONE);
        self.rank.resize(txs.len(), NONE);
        self.account_of.resize(txs.len(), NONE);
        output.clear();
        if output.capacity() < txs.len() {
            output.reserve(txs.len());
        }

        let mut stats = OrderStats {
            input_transactions: txs.len(),
            ..OrderStats::default()
        };

        for (index, tx) in txs.iter().enumerate() {
            let index = index as u32;

            if !TRUSTED_DENSE {
                if let Some(first_index) = self.hashes.find_or_insert_hash(txs, index) {
                    if txs[first_index as usize] != *tx {
                        return Err(OrderError::InconsistentDuplicateHash { hash: tx.hash });
                    }
                    stats.exact_duplicates += 1;
                    continue;
                }
                stats.unique_hashes += 1;
            }

            let account = if TRUSTED_DENSE {
                let account = dense_account_ids[index as usize];
                if account as usize >= self.base_nonces.len() {
                    return Err(OrderError::DenseAccountOutOfRange {
                        transaction_index: index as usize,
                        account_id: account,
                        account_count: self.base_nonces.len(),
                    });
                }
                account
            } else {
                self.account_index
                    .find_account(&self.accounts, &tx.sender)
                    .ok_or(OrderError::MissingAccountState { sender: tx.sender })?
            };
            if tx.nonce < self.base_nonces[account as usize] {
                stats.stale_transactions += 1;
                continue;
            }
            self.account_of[index as usize] = account;

            let probe = self
                .slots
                .probe_slot(txs, &self.account_of, index, account);
            match probe.existing {
                None => {
                    self.slots.write_probe(probe.bucket, index);
                    stats.selected_nonce_slots += 1;
                }
                Some(previous) => {
                    if TRUSTED_DENSE && tx.hash == txs[previous as usize].hash {
                        stats.exact_duplicates += 1;
                        continue;
                    }
                    stats.nonce_conflicts += 1;
                    if conflict_policy == ConflictPolicy::RejectBatch {
                        return Err(OrderError::ConflictingNonce {
                            sender: tx.sender,
                            nonce: tx.nonce,
                        });
                    }
                    if higher_priority(tx, &txs[previous as usize]) {
                        self.slots.write_probe(probe.bucket, index);
                    }
                }
            }
        }

        // Recover only the contiguous executable prefix of every sender chain.
        // This simultaneously identifies nonce-gap-blocked slots and records
        // direct successors, so the priority scheduling loop performs no hash
        // table lookups.
        for account_id in 0..self.base_nonces.len() {
            let account_id = account_id as u32;
            let mut current = self
                .slots
                .find_slot(
                    txs,
                    &self.account_of,
                    account_id,
                    self.base_nonces[account_id as usize],
                );
            if let Some(head) = current {
                self.heads.push(head);
            }

            while let Some(index) = current {
                self.reachable.push(index);
                let tx = &txs[index as usize];
                let next = tx
                    .nonce
                    .checked_add(1)
                    .and_then(|nonce| {
                        self.slots
                            .find_slot(txs, &self.account_of, account_id, nonce)
                    });
                self.successor[index as usize] = next.unwrap_or(NONE);
                current = next;
            }
        }

        stats.blocked_by_nonce_gap = stats.selected_nonce_slots - self.reachable.len();

        // With very few live account chains, scanning their current heads is
        // cheaper than globally ranking every transaction. The threshold is
        // intentionally small and can be tuned against the deployment CPU.
        if self.heads.len() <= LINEAR_READY_LIMIT {
            while !self.heads.is_empty() {
                let mut best_position = 0;
                for position in 1..self.heads.len() {
                    let candidate = self.heads[position];
                    let current_best = self.heads[best_position];
                    if higher_priority(
                        &txs[candidate as usize],
                        &txs[current_best as usize],
                    ) {
                        best_position = position;
                    }
                }

                let index = self.heads[best_position];
                output.push(index);
                let successor = self.successor[index as usize];
                if successor == NONE {
                    self.heads.swap_remove(best_position);
                } else {
                    self.heads[best_position] = successor;
                }
            }

            stats.ordered_transactions = output.len();
            debug_assert_eq!(stats.ordered_transactions, self.reachable.len());
            return Ok(stats);
        }

        // A fixed-leaf tournament avoids the O(r log r) global rank sort for
        // moderately many account chains. Replacing one leaf by its nonce
        // successor costs exactly one priority comparison per tree level.
        if self.heads.len() <= TOURNAMENT_READY_LIMIT {
            self.order_with_tournament(txs, output);
            stats.ordered_transactions = output.len();
            debug_assert_eq!(stats.ordered_transactions, self.reachable.len());
            return Ok(stats);
        }

        // Rank once by static priority. The ready set then stores ranks rather
        // than 64-byte keys and supports insert/extract-min in effectively O(1).
        self.rank_reachable(txs);

        // If every live account contributes one transaction, the priority rank
        // is already the complete answer and no ready-set traversal is needed.
        if self.reachable.len() == self.heads.len() {
            output.extend_from_slice(&self.reachable);
            stats.ordered_transactions = output.len();
            return Ok(stats);
        }

        for (rank, index) in self.reachable.iter().copied().enumerate() {
            self.rank[index as usize] = rank as u32;
        }

        self.ready.reset(self.reachable.len());
        for head in self.heads.iter().copied() {
            self.ready.insert(self.rank[head as usize] as usize);
        }

        while let Some(priority_rank) = self.ready.pop_min() {
            let index = self.reachable[priority_rank];
            output.push(index);
            let successor = self.successor[index as usize];
            if successor != NONE {
                self.ready.insert(self.rank[successor as usize] as usize);
            }
        }

        stats.ordered_transactions = output.len();
        debug_assert_eq!(stats.ordered_transactions, self.reachable.len());
        Ok(stats)
    }

    fn order_with_tournament(&mut self, txs: &[TxMeta], output: &mut Vec<u32>) {
        debug_assert!(!self.heads.is_empty());
        let leaf_base = self.heads.len().next_power_of_two();
        self.tournament.resize(leaf_base * 2, NONE);
        self.tournament.fill(NONE);

        for leaf in 0..self.heads.len() {
            self.tournament[leaf_base + leaf] = leaf as u32;
        }
        for node in (1..leaf_base).rev() {
            self.tournament[node] = tournament_winner(
                self.tournament[node << 1],
                self.tournament[(node << 1) | 1],
                &self.heads,
                txs,
            );
        }

        loop {
            let leaf = self.tournament[1];
            if leaf == NONE {
                break;
            }
            let leaf = leaf as usize;
            let index = self.heads[leaf];
            output.push(index);

            let successor = self.successor[index as usize];
            self.heads[leaf] = successor;
            self.tournament[leaf_base + leaf] = if successor == NONE {
                NONE
            } else {
                leaf as u32
            };

            let mut node = (leaf_base + leaf) >> 1;
            while node != 0 {
                self.tournament[node] = tournament_winner(
                    self.tournament[node << 1],
                    self.tournament[(node << 1) | 1],
                    &self.heads,
                    txs,
                );
                node >>= 1;
            }
        }
    }

    fn rank_reachable(&mut self, txs: &[TxMeta]) {
        let tips_fit_u64 = self.reachable.iter().all(|index| {
            let limbs = txs[*index as usize].effective_tip.limbs;
            (limbs[1] | limbs[2] | limbs[3]) == 0
        });

        if tips_fit_u64 {
            self.rank64.clear();
            for index in self.reachable.iter().copied() {
                let tx = &txs[index as usize];
                self.rank64.push(Rank64 {
                    tip: tx.effective_tip.limbs[0],
                    hash_prefix: hash_prefix(&tx.hash),
                    index,
                    _padding: 0,
                });
            }
            self.rank64.sort_unstable_by(|left, right| {
                right
                    .tip
                    .cmp(&left.tip)
                    .then_with(|| left.hash_prefix.cmp(&right.hash_prefix))
                    .then_with(|| {
                        txs[left.index as usize]
                            .hash
                            .cmp(&txs[right.index as usize].hash)
                    })
                    .then_with(|| left.index.cmp(&right.index))
            });
            for (target, record) in self.reachable.iter_mut().zip(&self.rank64) {
                *target = record.index;
            }
        } else {
            self.rank256.clear();
            for index in self.reachable.iter().copied() {
                let tx = &txs[index as usize];
                self.rank256.push(Rank256 {
                    tip: tx.effective_tip,
                    hash_prefix: hash_prefix(&tx.hash),
                    index,
                    _padding: 0,
                });
            }
            self.rank256.sort_unstable_by(|left, right| {
                right
                    .tip
                    .cmp(&left.tip)
                    .then_with(|| left.hash_prefix.cmp(&right.hash_prefix))
                    .then_with(|| {
                        txs[left.index as usize]
                            .hash
                            .cmp(&txs[right.index as usize].hash)
                    })
                    .then_with(|| left.index.cmp(&right.index))
            });
            for (target, record) in self.reachable.iter_mut().zip(&self.rank256) {
                *target = record.index;
            }
        }
    }
}

#[inline(always)]
fn priority_order(left: &TxMeta, right: &TxMeta) -> Ordering {
    right
        .effective_tip
        .cmp(&left.effective_tip)
        .then_with(|| left.hash.cmp(&right.hash))
}

#[inline(always)]
fn higher_priority(left: &TxMeta, right: &TxMeta) -> bool {
    priority_order(left, right) == Ordering::Less
}

#[inline(always)]
fn tournament_winner(
    left_leaf: u32,
    right_leaf: u32,
    heads: &[u32],
    txs: &[TxMeta],
) -> u32 {
    if left_leaf == NONE {
        return right_leaf;
    }
    if right_leaf == NONE {
        return left_leaf;
    }
    let left = heads[left_leaf as usize];
    let right = heads[right_leaf as usize];
    debug_assert_ne!(left, NONE);
    debug_assert_ne!(right, NONE);
    if higher_priority(&txs[left as usize], &txs[right as usize]) {
        left_leaf
    } else {
        right_leaf
    }
}

#[inline]
fn check_batch_size(count: usize) -> Result<(), OrderError> {
    if count > u32::MAX as usize {
        Err(OrderError::TooManyTransactions { count })
    } else {
        Ok(())
    }
}

#[inline]
fn reserve_to<T>(buffer: &mut Vec<T>, capacity: usize) {
    if buffer.capacity() < capacity {
        buffer.reserve(capacity - buffer.len());
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Rank64 {
    tip: u64,
    hash_prefix: u64,
    index: u32,
    _padding: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Rank256 {
    tip: Tip256,
    hash_prefix: u64,
    index: u32,
    _padding: u32,
}

#[inline(always)]
fn hash_prefix(hash: &TxHash) -> u64 {
    u64::from_be_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ])
}

#[derive(Clone, Copy)]
struct HashSeed {
    first: u64,
    second: u64,
}

impl HashSeed {
    fn random() -> Self {
        // RandomState obtains per-process secret material from the platform.
        // We pay for SipHash only here, outside the hot path, then use the
        // resulting secrets with the fast fixed-width mixer below.
        let state = RandomState::new();
        let mut first = state.build_hasher();
        first.write_u64(0x6a09_e667_f3bc_c909);
        let mut second = state.build_hasher();
        second.write_u64(0xbb67_ae85_84ca_a73b);
        Self {
            first: first.finish(),
            second: second.finish(),
        }
    }

    #[inline(always)]
    fn derive(self, domain: u64) -> Self {
        Self {
            first: mix64(self.first ^ domain),
            second: mix64(self.second ^ domain.rotate_left(29)),
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Bucket {
    epoch: u32,
    value_plus_one: u32,
}

struct SlotProbe {
    bucket: usize,
    existing: Option<u32>,
}

/// An epoch-cleared open-addressing table. Values are indices into caller-owned
/// arrays. A load factor no greater than one half keeps linear probing short and
/// makes reset O(1) except once per 2^32 batches.
struct IndexTable {
    buckets: Vec<Bucket>,
    mask: usize,
    epoch: u32,
    seed: HashSeed,
}

impl IndexTable {
    const fn new(seed: HashSeed) -> Self {
        Self {
            buckets: Vec::new(),
            mask: 0,
            epoch: 0,
            seed,
        }
    }

    fn ensure_capacity(&mut self, entries: usize) -> Result<(), OrderError> {
        check_batch_size(entries)?;
        let doubled = entries
            .checked_mul(2)
            .ok_or(OrderError::TooManyTransactions { count: entries })?;
        let required = doubled
            .max(8)
            .checked_next_power_of_two()
            .ok_or(OrderError::TooManyTransactions { count: entries })?;

        if self.buckets.len() < required {
            self.buckets.resize(required, Bucket::default());
            // `resize` preserves the old prefix. Since growth restarts the
            // epoch sequence below, those old tags must not become live again.
            self.buckets.fill(Bucket::default());
            self.mask = required - 1;
            self.epoch = 0;
        }
        Ok(())
    }

    fn start_batch(&mut self, entries: usize) -> Result<(), OrderError> {
        self.ensure_capacity(entries)?;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.buckets.fill(Bucket::default());
            self.epoch = 1;
        }
        Ok(())
    }

    #[inline(always)]
    fn find_or_insert_hash(&mut self, txs: &[TxMeta], index: u32) -> Option<u32> {
        let hash = &txs[index as usize].hash;
        let mut bucket = hash_tx_hash(hash, self.seed) & self.mask;
        loop {
            let entry = &mut self.buckets[bucket];
            if entry.epoch != self.epoch {
                entry.epoch = self.epoch;
                entry.value_plus_one = index + 1;
                return None;
            }
            let existing = entry.value_plus_one - 1;
            if txs[existing as usize].hash == *hash {
                return Some(existing);
            }
            bucket = (bucket + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn probe_slot(
        &mut self,
        txs: &[TxMeta],
        account_of: &[u32],
        index: u32,
        account: u32,
    ) -> SlotProbe {
        let tx = &txs[index as usize];
        let mut bucket = hash_slot(account, tx.nonce, self.seed) & self.mask;
        loop {
            let entry = self.buckets[bucket];
            if entry.epoch != self.epoch {
                return SlotProbe {
                    bucket,
                    existing: None,
                };
            }
            let existing = entry.value_plus_one - 1;
            let old = &txs[existing as usize];
            if old.nonce == tx.nonce && account_of[existing as usize] == account {
                return SlotProbe {
                    bucket,
                    existing: Some(existing),
                };
            }
            bucket = (bucket + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn write_probe(&mut self, bucket: usize, index: u32) {
        self.buckets[bucket] = Bucket {
            epoch: self.epoch,
            value_plus_one: index + 1,
        };
    }

    #[inline(always)]
    fn find_slot(
        &self,
        txs: &[TxMeta],
        account_of: &[u32],
        account: u32,
        nonce: u64,
    ) -> Option<u32> {
        let mut bucket = hash_slot(account, nonce, self.seed) & self.mask;
        loop {
            let entry = self.buckets[bucket];
            if entry.epoch != self.epoch {
                return None;
            }
            let existing = entry.value_plus_one - 1;
            let tx = &txs[existing as usize];
            if tx.nonce == nonce && account_of[existing as usize] == account {
                return Some(existing);
            }
            bucket = (bucket + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn find_account(&self, accounts: &[AccountNonce], sender: &Address) -> Option<u32> {
        let mut bucket = hash_address(sender, self.seed) & self.mask;
        loop {
            let entry = self.buckets[bucket];
            if entry.epoch != self.epoch {
                return None;
            }
            let existing = entry.value_plus_one - 1;
            if accounts[existing as usize].sender == *sender {
                return Some(existing);
            }
            bucket = (bucket + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn insert_account(&mut self, accounts: &[AccountNonce], index: u32) {
        let sender = &accounts[index as usize].sender;
        let mut bucket = hash_address(sender, self.seed) & self.mask;
        loop {
            let entry = &mut self.buckets[bucket];
            if entry.epoch != self.epoch {
                entry.epoch = self.epoch;
                entry.value_plus_one = index + 1;
                return;
            }
            bucket = (bucket + 1) & self.mask;
        }
    }
}

#[inline(always)]
fn hash_tx_hash(hash: &TxHash, seed: HashSeed) -> usize {
    let a = load_u64_le(hash, 0);
    let b = load_u64_le(hash, 8);
    let c = load_u64_le(hash, 16);
    let d = load_u64_le(hash, 24);
    mix64(
        mix64(a ^ seed.first)
            ^ b.rotate_left(17)
            ^ c.rotate_left(33)
            ^ d.rotate_left(49)
            ^ seed.second,
    ) as usize
}

#[inline(always)]
fn hash_address(address: &Address, seed: HashSeed) -> usize {
    let a = load_u64_le(address, 0);
    let b = load_u64_le(address, 8);
    let c = u32::from_le_bytes([address[16], address[17], address[18], address[19]]) as u64;
    mix64(
        mix64(a ^ seed.first) ^ b.rotate_left(23) ^ c.rotate_left(47) ^ seed.second,
    ) as usize
}

#[inline(always)]
fn hash_slot(account: u32, nonce: u64, seed: HashSeed) -> usize {
    let account_multiplier = seed.first | 1;
    let nonce_multiplier = seed.second | 1;
    mix64(
        (account as u64).wrapping_mul(account_multiplier)
            ^ nonce.wrapping_mul(nonce_multiplier)
            ^ seed.first.rotate_left(17)
            ^ seed.second.rotate_left(41),
    ) as usize
}

#[inline(always)]
fn load_u64_le<const N: usize>(bytes: &[u8; N], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

#[inline(always)]
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// A min-set over integer ranks. Level zero stores rank bits and every higher
/// level summarizes nonempty words in the level below.
struct HierarchicalBitSet {
    levels: Vec<Vec<u64>>,
    active_levels: usize,
}

impl HierarchicalBitSet {
    const fn new() -> Self {
        Self {
            levels: Vec::new(),
            active_levels: 0,
        }
    }

    fn ensure_universe(&mut self, universe: usize) {
        if universe == 0 {
            return;
        }
        let mut words = universe.div_ceil(64);
        let mut level = 0;
        loop {
            if self.levels.len() <= level {
                self.levels.push(Vec::new());
            }
            reserve_to(&mut self.levels[level], words);
            if words == 1 {
                break;
            }
            words = words.div_ceil(64);
            level += 1;
        }
    }

    fn reset(&mut self, universe: usize) {
        if universe == 0 {
            self.active_levels = 0;
            return;
        }
        self.ensure_universe(universe);
        let mut words = universe.div_ceil(64);
        let mut used_levels = 0;
        loop {
            self.levels[used_levels].resize(words, 0);
            self.levels[used_levels].fill(0);
            used_levels += 1;
            if words == 1 {
                break;
            }
            words = words.div_ceil(64);
        }
        self.active_levels = used_levels;
    }

    #[inline(always)]
    fn insert(&mut self, mut bit_index: usize) {
        for level in self.levels.iter_mut().take(self.active_levels) {
            let word_index = bit_index >> 6;
            let mask = 1u64 << (bit_index & 63);
            let word = &mut level[word_index];
            let was_empty = *word == 0;
            *word |= mask;
            if !was_empty {
                break;
            }
            bit_index = word_index;
        }
    }

    #[inline(always)]
    fn pop_min(&mut self) -> Option<usize> {
        if self.active_levels == 0 {
            return None;
        }
        let top = &self.levels[self.active_levels - 1];
        let top_word = top[0];
        if top_word == 0 {
            return None;
        }

        let mut index = top_word.trailing_zeros() as usize;
        for level_index in (0..self.active_levels - 1).rev() {
            let word = self.levels[level_index][index];
            debug_assert_ne!(word, 0);
            index = (index << 6) | word.trailing_zeros() as usize;
        }
        self.remove(index);
        Some(index)
    }

    #[inline(always)]
    fn remove(&mut self, mut bit_index: usize) {
        for level in self.levels.iter_mut().take(self.active_levels) {
            let word_index = bit_index >> 6;
            let mask = 1u64 << (bit_index & 63);
            let word = &mut level[word_index];
            *word &= !mask;
            if *word != 0 {
                break;
            }
            bit_index = word_index;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::{BTreeSet, HashMap, HashSet};

    fn address(id: u8) -> Address {
        let mut out = [0; 20];
        out[19] = id;
        out
    }

    fn wide_address(id: u64) -> Address {
        let mut out = [0; 20];
        out[12..].copy_from_slice(&id.to_be_bytes());
        out
    }

    fn hash(id: u64) -> TxHash {
        let mut out = [0; 32];
        out[24..].copy_from_slice(&id.to_be_bytes());
        out
    }

    fn tx(sender: u8, nonce: u64, tip: u64, hash_id: u64) -> TxMeta {
        TxMeta {
            effective_tip: Tip256::from_u64(tip),
            hash: hash(hash_id),
            sender: address(sender),
            nonce,
        }
    }

    fn oracle_higher_priority(left: &TxMeta, right: &TxMeta) -> bool {
        for limb in (0..4).rev() {
            if left.effective_tip.limbs[limb] != right.effective_tip.limbs[limb] {
                return left.effective_tip.limbs[limb] > right.effective_tip.limbs[limb];
            }
        }
        left.hash < right.hash
    }

    #[test]
    fn priority_kahn_respects_nonce_dependencies() {
        let txs = vec![
            tx(1, 0, 1, 1),
            tx(1, 1, 100, 2),
            tx(2, 0, 50, 3),
            tx(2, 1, 2, 4),
        ];
        let state = [
            AccountNonce {
                sender: address(1),
                next_nonce: 0,
            },
            AccountNonce {
                sender: address(2),
                next_nonce: 0,
            },
        ];
        let mut orderer = Orderer::with_capacity(16).unwrap();
        let mut out = Vec::new();
        let stats = orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![2, 3, 0, 1]);
        assert_eq!(stats.ordered_transactions, 4);
    }

    #[test]
    fn newly_unlocked_high_tip_transaction_preempts_other_chain() {
        let txs = vec![
            tx(1, 0, 5, 1),
            tx(1, 1, 1, 2),
            tx(2, 0, 4, 3),
            tx(2, 1, 10, 4),
        ];
        let state = [
            AccountNonce {
                sender: address(1),
                next_nonce: 0,
            },
            AccountNonce {
                sender: address(2),
                next_nonce: 0,
            },
        ];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![0, 2, 3, 1]);
    }

    #[test]
    fn hash_breaks_ready_tip_ties_ascending() {
        let txs = vec![tx(1, 0, 7, 9), tx(2, 0, 7, 3), tx(3, 0, 7, 5)];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        orderer
            .order_assuming_min_nonce(&txs, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![1, 2, 0]);
    }

    #[test]
    fn removes_exact_duplicates_and_selects_best_nonce_conflict() {
        let winner = tx(1, 4, 20, 10);
        let loser = tx(1, 4, 10, 11);
        let next = tx(1, 5, 1, 12);
        let txs = vec![loser, winner, winner, next];
        let state = [AccountNonce {
            sender: address(1),
            next_nonce: 4,
        }];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        let stats = orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![1, 3]);
        assert_eq!(stats.exact_duplicates, 1);
        assert_eq!(stats.nonce_conflicts, 1);
        assert_eq!(stats.selected_nonce_slots, 2);
    }

    #[test]
    fn reports_stale_and_gap_blocked_transactions() {
        let txs = vec![
            tx(1, 4, 100, 1),
            tx(1, 5, 3, 2),
            tx(1, 7, 100, 3),
            tx(1, 8, 100, 4),
        ];
        let state = [AccountNonce {
            sender: address(1),
            next_nonce: 5,
        }];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        let stats = orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![1]);
        assert_eq!(stats.stale_transactions, 1);
        assert_eq!(stats.blocked_by_nonce_gap, 2);
    }

    #[test]
    fn strict_mode_rejects_same_nonce_conflicts() {
        let txs = vec![tx(1, 0, 1, 1), tx(1, 0, 2, 2)];
        let mut orderer = Orderer::new();
        let error = orderer
            .order_assuming_min_nonce(&txs, &mut Vec::new(), ConflictPolicy::RejectBatch)
            .unwrap_err();
        assert!(matches!(error, OrderError::ConflictingNonce { nonce: 0, .. }));
    }

    #[test]
    fn conflict_tip_tie_uses_lower_hash() {
        let txs = vec![tx(1, 0, 9, 10), tx(1, 0, 9, 2), tx(1, 1, 1, 20)];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        orderer
            .order_assuming_min_nonce(&txs, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn hash_prefix_collision_uses_all_256_bits() {
        let mut first = tx(1, 0, 5, 0);
        let mut second = tx(2, 0, 5, 0);
        first.hash = [0x11; 32];
        second.hash = [0x11; 32];
        first.hash[31] = 1;
        second.hash[31] = 2;
        let txs = [second, first];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        orderer
            .order_assuming_min_nonce(&txs, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![1, 0]);
    }

    #[test]
    fn full_u256_tip_ranking_path_is_exact() {
        let mut txs = Vec::new();
        for sender in 0..10u8 {
            let mut transaction = tx(sender, 0, 0, sender as u64);
            transaction.effective_tip = Tip256 {
                limbs: [sender as u64, 7, (sender % 3) as u64, 1],
            };
            txs.push(transaction);
        }
        let mut expected: Vec<u32> = (0..txs.len() as u32).collect();
        expected.sort_unstable_by(|left, right| {
            let left_tx = &txs[*left as usize];
            let right_tx = &txs[*right as usize];
            if oracle_higher_priority(left_tx, right_tx) {
                Ordering::Less
            } else if oracle_higher_priority(right_tx, left_tx) {
                Ordering::Greater
            } else {
                left.cmp(right)
            }
        });

        let mut orderer = Orderer::new();
        orderer.reachable = (0..txs.len() as u32).rev().collect();
        orderer.rank_reachable(&txs);
        assert_eq!(orderer.reachable, expected);
    }

    #[test]
    fn u256_tip_compares_most_significant_limbs_first() {
        let high = Tip256 {
            limbs: [0, 0, 0, 1],
        };
        let low = Tip256 {
            limbs: [u64::MAX, u64::MAX, u64::MAX, 0],
        };
        assert!(high > low);
    }

    #[test]
    fn tournament_path_matches_oracle() {
        let mut txs = Vec::new();
        let mut state = Vec::new();
        for sender in 0..12u8 {
            state.push(AccountNonce {
                sender: address(sender),
                next_nonce: 0,
            });
            txs.push(tx(sender, 0, (sender as u64 * 7) % 13, sender as u64));
            txs.push(tx(
                sender,
                1,
                100 - sender as u64,
                1_000 + sender as u64,
            ));
        }
        txs.reverse();
        let expected = oracle(&txs, &state);
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn hierarchical_bitset_crosses_word_and_summary_boundaries() {
        let mut set = HierarchicalBitSet::new();
        set.reset(10_000);
        let inserted = [
            9_999usize, 4_097, 4_096, 4_095, 65, 64, 63, 1, 0, 8_191, 8_192,
        ];
        for rank in inserted {
            set.insert(rank);
        }
        let mut actual = Vec::new();
        while let Some(rank) = set.pop_min() {
            actual.push(rank);
        }
        let mut expected = inserted.to_vec();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn rank_and_multilevel_ready_set_integration_matches_btree_oracle() {
        const SENDERS: usize = TOURNAMENT_READY_LIMIT + 1;
        let mut txs = Vec::with_capacity(SENDERS * 2);
        let mut state = Vec::with_capacity(SENDERS);
        for sender in 0..SENDERS {
            let address = wide_address(sender as u64);
            state.push(AccountNonce {
                sender: address,
                next_nonce: 0,
            });
            txs.push(TxMeta {
                effective_tip: Tip256::from_u64((sender % 97) as u64),
                hash: hash((sender * 2) as u64),
                sender: address,
                nonce: 0,
            });
            txs.push(TxMeta {
                effective_tip: Tip256::from_u64((10_000 - sender) as u64),
                hash: hash((sender * 2 + 1) as u64),
                sender: address,
                nonce: 1,
            });
        }
        txs.reverse();

        let mut slot = HashMap::new();
        for (index, tx) in txs.iter().enumerate() {
            slot.insert((tx.sender, tx.nonce), index as u32);
        }
        let mut ready: BTreeSet<(Reverse<u64>, TxHash, u32)> = BTreeSet::new();
        for account in &state {
            let index = slot[&(account.sender, account.next_nonce)];
            let tx = &txs[index as usize];
            ready.insert((Reverse(tx.effective_tip.limbs[0]), tx.hash, index));
        }
        let mut expected = Vec::with_capacity(txs.len());
        while let Some(key) = ready.iter().next().cloned() {
            ready.remove(&key);
            let index = key.2;
            expected.push(index);
            let tx = &txs[index as usize];
            if let Some(next_nonce) = tx.nonce.checked_add(1) {
                if let Some(&next) = slot.get(&(tx.sender, next_nonce)) {
                    let next_tx = &txs[next as usize];
                    ready.insert((Reverse(next_tx.effective_tip.limbs[0]), next_tx.hash, next));
                }
            }
        }

        let mut orderer = Orderer::with_capacity(txs.len()).unwrap();
        let mut actual = Vec::with_capacity(txs.len());
        orderer
            .order_with_state(
                &txs,
                &state,
                &mut actual,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn maximum_nonce_executes_once_without_wrapping() {
        let txs = [tx(1, u64::MAX, 9, 1), tx(1, 0, 100, 2)];
        let state = [AccountNonce {
            sender: address(1),
            next_nonce: u64::MAX,
        }];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        let stats = orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out, vec![0]);
        assert_eq!(stats.stale_transactions, 1);
    }

    #[test]
    fn missing_state_and_inconsistent_hash_are_errors() {
        let transaction = tx(1, 0, 1, 1);
        let mut orderer = Orderer::new();
        let missing = orderer
            .order_with_state(
                &[transaction],
                &[],
                &mut Vec::new(),
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap_err();
        assert!(matches!(missing, OrderError::MissingAccountState { .. }));

        let mut inconsistent = transaction;
        inconsistent.nonce = 1;
        let mismatch = orderer
            .order_assuming_min_nonce(
                &[transaction, inconsistent],
                &mut Vec::new(),
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap_err();
        assert!(matches!(
            mismatch,
            OrderError::InconsistentDuplicateHash { .. }
        ));
    }

    #[test]
    fn trusted_dense_path_matches_checked_path() {
        let txs = vec![
            tx(0, 0, 3, 1),
            tx(0, 0, 3, 1),
            tx(0, 1, 20, 2),
            tx(1, 0, 4, 3),
            tx(1, 0, 2, 4),
            tx(1, 1, 30, 5),
            tx(2, 2, 100, 6),
        ];
        let states = [
            AccountNonce {
                sender: address(0),
                next_nonce: 0,
            },
            AccountNonce {
                sender: address(1),
                next_nonce: 0,
            },
            AccountNonce {
                sender: address(2),
                next_nonce: 0,
            },
        ];
        let ids = [0, 0, 0, 1, 1, 1, 2];
        let bases = [0, 0, 0];
        let mut orderer = Orderer::new();
        let mut checked = Vec::new();
        orderer
            .order_with_state(
                &txs,
                &states,
                &mut checked,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        let mut dense = Vec::new();
        let count = orderer
            .order_trusted_dense(
                &txs,
                &ids,
                &bases,
                &mut dense,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        assert_eq!(dense, checked);
        assert_eq!(count, checked.len());
    }

    #[test]
    fn trusted_dense_path_validates_shape_and_range() {
        let transaction = [tx(0, 0, 1, 1)];
        let mut orderer = Orderer::new();
        let mismatch = orderer
            .order_trusted_dense(
                &transaction,
                &[],
                &[0],
                &mut Vec::new(),
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap_err();
        assert!(matches!(
            mismatch,
            OrderError::DenseAccountLengthMismatch { .. }
        ));

        let out_of_range = orderer
            .order_trusted_dense(
                &transaction,
                &[1],
                &[0],
                &mut Vec::new(),
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap_err();
        assert!(matches!(
            out_of_range,
            OrderError::DenseAccountOutOfRange { .. }
        ));
    }

    #[test]
    fn workspace_reuse_across_large_and_empty_batches_is_clean() {
        let mut orderer = Orderer::with_capacity(100_000).unwrap();
        let mut txs = Vec::with_capacity(10_000);
        let mut state = Vec::with_capacity(10_000);
        for id in 0..10_000u64 {
            let sender = (id % 251) as u8;
            txs.push(tx(sender, id / 251, id % 17, id));
        }
        for sender in 0..251u16 {
            state.push(AccountNonce {
                sender: address(sender as u8),
                next_nonce: 0,
            });
        }
        let mut out = Vec::with_capacity(100_000);
        orderer
            .order_with_state(&txs, &state, &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert_eq!(out.len(), txs.len());

        let empty = orderer
            .order_with_state(&[], &[], &mut out, ConflictPolicy::KeepHigherPriority)
            .unwrap();
        assert!(out.is_empty());
        assert_eq!(empty, OrderStats::default());

        let single = [tx(7, 3, 4, 999_999)];
        orderer
            .order_assuming_min_nonce(
                &single,
                &mut out,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn workspace_growth_after_use_cannot_revive_old_buckets() {
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        let first = [tx(1, 0, 2, 1)];
        orderer
            .order_assuming_min_nonce(
                &first,
                &mut out,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        assert_eq!(out, vec![0]);

        let mut second = Vec::new();
        for sender in 0..100u8 {
            second.push(tx(sender, 0, sender as u64, 10_000 + sender as u64));
        }
        orderer
            .order_assuming_min_nonce(
                &second,
                &mut out,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        assert_eq!(out, (0..100u32).rev().collect::<Vec<_>>());
    }

    #[test]
    fn index_table_growth_clears_preserved_epoch_tags() {
        let seed = HashSeed {
            first: 1,
            second: 2,
        };
        let mut table = IndexTable::new(seed);
        let txs = [tx(1, 0, 1, 1)];
        table.start_batch(1).unwrap();
        assert_eq!(table.find_or_insert_hash(&txs, 0), None);
        assert!(table.buckets.iter().any(|bucket| bucket.epoch == 1));

        table.ensure_capacity(100).unwrap();
        assert!(table
            .buckets
            .iter()
            .all(|bucket| bucket.epoch == 0 && bucket.value_plus_one == 0));
    }

    #[test]
    fn index_table_epoch_wrap_performs_full_clear() {
        let seed = HashSeed {
            first: 3,
            second: 5,
        };
        let mut table = IndexTable::new(seed);
        table.ensure_capacity(8).unwrap();
        table.buckets[0] = Bucket {
            epoch: 17,
            value_plus_one: 99,
        };
        table.epoch = u32::MAX;
        table.start_batch(8).unwrap();
        assert_eq!(table.epoch, 1);
        assert!(table
            .buckets
            .iter()
            .all(|bucket| bucket.epoch == 0 && bucket.value_plus_one == 0));
    }

    #[test]
    fn duplicate_account_state_must_be_consistent() {
        let sender = address(1);
        let txs = [tx(1, 4, 1, 1)];
        let mut orderer = Orderer::new();
        let mut out = Vec::new();
        orderer
            .order_with_state(
                &txs,
                &[
                    AccountNonce {
                        sender,
                        next_nonce: 4,
                    },
                    AccountNonce {
                        sender,
                        next_nonce: 4,
                    },
                ],
                &mut out,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap();
        assert_eq!(out, vec![0]);

        let error = orderer
            .order_with_state(
                &txs,
                &[
                    AccountNonce {
                        sender,
                        next_nonce: 4,
                    },
                    AccountNonce {
                        sender,
                        next_nonce: 5,
                    },
                ],
                &mut out,
                ConflictPolicy::KeepHigherPriority,
            )
            .unwrap_err();
        assert!(matches!(error, OrderError::ConflictingAccountState { .. }));
    }

    fn oracle(txs: &[TxMeta], states: &[AccountNonce]) -> Vec<u32> {
        let mut next_nonce: HashMap<Address, u64> = states
            .iter()
            .map(|state| (state.sender, state.next_nonce))
            .collect();
        let mut seen = HashSet::new();
        let mut slots: HashMap<(Address, u64), u32> = HashMap::new();

        for (index, tx) in txs.iter().enumerate() {
            if !seen.insert(tx.hash) {
                continue;
            }
            if tx.nonce < next_nonce[&tx.sender] {
                continue;
            }
            slots
                .entry((tx.sender, tx.nonce))
                .and_modify(|old| {
                    if oracle_higher_priority(tx, &txs[*old as usize]) {
                        *old = index as u32;
                    }
                })
                .or_insert(index as u32);
        }

        let mut out = Vec::new();
        loop {
            let mut best: Option<u32> = None;
            for (&(sender, nonce), &index) in &slots {
                if next_nonce.get(&sender).copied() != Some(nonce) {
                    continue;
                }
                if best
                    .map(|old| {
                        oracle_higher_priority(&txs[index as usize], &txs[old as usize])
                    })
                    .unwrap_or(true)
                {
                    best = Some(index);
                }
            }
            let Some(index) = best else { break };
            out.push(index);
            let tx = &txs[index as usize];
            if let Some(next) = tx.nonce.checked_add(1) {
                *next_nonce.get_mut(&tx.sender).unwrap() = next;
            } else {
                next_nonce.remove(&tx.sender);
            }
            slots.remove(&(tx.sender, tx.nonce));
        }
        out
    }

    #[test]
    fn randomized_small_batches_match_quadratic_oracle() {
        let mut random = SplitMix64(0x6a09_e667_f3bc_c909);
        let mut orderer = Orderer::with_capacity(128).unwrap();
        let mut actual = Vec::new();
        let mut dense_actual = Vec::new();

        for case in 0..1_000u64 {
            let sender_count = (random.next() % 16 + 1) as u8;
            let mut states = Vec::new();
            for sender in 0..sender_count {
                states.push(AccountNonce {
                    sender: address(sender),
                    next_nonce: random.next() % 3,
                });
            }

            let count = (random.next() % 100) as usize;
            let mut txs = Vec::with_capacity(count + count / 8);
            for index in 0..count {
                let sender = (random.next() % sender_count as u64) as u8;
                let base = states[sender as usize].next_nonce;
                let transaction = tx(
                    sender,
                    base + random.next() % 6,
                    random.next() % 10,
                    (case << 32) | index as u64,
                );
                txs.push(transaction);
                if random.next() & 15 == 0 {
                    txs.push(transaction);
                }
            }

            let expected = oracle(&txs, &states);
            orderer
                .order_with_state(
                    &txs,
                    &states,
                    &mut actual,
                    ConflictPolicy::KeepHigherPriority,
                )
                .unwrap();
            assert_eq!(actual, expected, "case {case}");

            let account_ids: Vec<u32> = txs
                .iter()
                .map(|transaction| transaction.sender[19] as u32)
                .collect();
            let bases: Vec<u64> = states.iter().map(|state| state.next_nonce).collect();
            orderer
                .order_trusted_dense(
                    &txs,
                    &account_ids,
                    &bases,
                    &mut dense_actual,
                    ConflictPolicy::KeepHigherPriority,
                )
                .unwrap();
            assert_eq!(dense_actual, expected, "dense case {case}");
        }
    }

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            mix64(self.0)
        }
    }
}
