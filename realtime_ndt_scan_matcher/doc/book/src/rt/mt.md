# The `mt` multi-core engine

The `mt` feature is the multi-core `no_std` (kernel) configuration: the engine stays `Sync` and
shareable across cores without `std` or an external mutex.

## Interrupt-safe cells

The interior-mutability cells become `awkernel_sync::Mutex<Arc<…>>`, whose guards **disable
interrupts** while held. So every critical section is a few instructions — an `Arc` refcount bump
(load) or a pointer swap (store) — **never** an align, a deep clone, or the drop of an old map.

## rcu without a long lock

`swap_rcu` under `mt` runs the update closure **outside** the lock (it may deep-clone a whole map,
which must never happen with interrupts disabled) and publishes with an optimistic `Arc::ptr_eq`
retry; old snapshots also drop outside the lock. Writers (params setup, map commit) are rare, so it
converges immediately. See [Concurrency](../arch/concurrency.md).

## Caller-owned scratch

There is no engine-owned or thread-local scratch in any configuration. Each task or thread owns a
[`MatchScratch`](../arch/scratch.md) and passes it to every alignment call, so cross-call mutable
scratch dependencies are explicit in the type system.

## Building it

`--no-default-features --features mt`. `awkernel_sync` is a pinned git revision (recorded in
`Cargo.lock`) with `default-features = false`, so the **final binary** must enable exactly one
backend feature (`x86`/`aarch64`/`rv64`/`rv32` on bare metal, or `awkernel_sync/std` on a host) —
otherwise `awkernel_sync` does not compile. `examples/threads_ndt.rs` exercises shared-engine usage;
the host-testable lint gate is `--features mt,awkernel_sync/std` (see
Build and test).

> Source: `src/engine.rs` (`mt` cfg paths), `Cargo.toml` (`mt` feature), `examples/threads_ndt.rs`.
