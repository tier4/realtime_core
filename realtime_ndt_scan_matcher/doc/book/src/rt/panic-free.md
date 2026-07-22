# Panic-free, bounded execution

The crate's lint hardening, seen through a real-time lens: the align path has no hidden abort, no
unbounded work, and no allocation.

## No panic source

Non-test code bans `unwrap` / `expect` / `panic` / `unreachable` / `todo` / `unimplemented` and
`indexing_slicing` / `string_slice` — so there is no implicit panic (which would abort or unwind) on
the RT path. Fallible operations return `Result`/`Option` and are handled; genuinely-dynamic
indexing uses `.get()`/iterators.

## No silent overflow

`[profile.release] overflow-checks = true` makes an integer overflow a **panic**, so integer
arithmetic must be explicit `checked_*` / `saturating_*` / `wrapping_*` — never bare `+`/`-`/`*`.
Float math (which does not panic) uses a scoped, documented allowance in the numeric kernels only.
Lossy `as` casts are banned in favor of `TryFrom`; a `Result` is never discarded with `let _`.

**Standalone-consumer caveat:** `overflow-checks = true` lives in the **workspace root** manifest
(the node crate) and covers both members for every in-tree build (colcon/Corrosion and
`cd engine && cargo build --release` alike). Cargo profiles do **not** travel with a dependency,
however — a consumer that imports the engine crate from outside this workspace (e.g. the `no_std`
kernel repo) gets **its own** profile, so that final binary must re-declare
`overflow-checks = true` (and re-apply the lint gates in its CI) to keep this guarantee. See the
note in `Cargo.toml`.

## Bounded and non-blocking

On the align path: loops are statically bounded (`max_iterations`, `MAX_NEIGHBORS`), buffers are
pre-sized and reused ([zero-alloc](zero-alloc.md)), and there is no blocking, no logging/formatting,
and no user callback. Together these give the [WCET contract](wcet.md) its teeth.

## Review

A `rust-realtime-review` accompanies each engine/align patch to keep these properties from
regressing; the lints themselves are enforced in CI (see
Lint gates and suppression policy).

> Source: `Cargo.toml` `[lints]` / `[profile.release]`; `src/ndt.rs`.
