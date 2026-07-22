# Map update

How the target map is refreshed at runtime without disturbing concurrent alignment.

## Policy and state (Rust-owned)

`MapUpdateState` holds the map-update decision state: `last_update_position`, whether the next update
must `need_rebuild`, and the loaded cell ids live in the engine. Rust decides *when* an update is
needed (first update, and whenever the vehicle has moved far enough that the loaded map can no longer
cover the LiDAR range) and whether a full rebuild is required.

## The atomic commit

`scan_matcher::apply_map_update` performs:

1. `source.load(center, radius).await` — fetch the delta through the async `MapSource`
   port (empty delta ⇒ no-op, no republish).
2. Build the new map on a **private staging engine** — `engine.clone()` for an incremental update, or
   `engine.clone_empty()` for a rebuild — apply the added/removed tiles, and `create_kdtree()`.
3. `engine.commit_from(&staging)` — one atomic store publishes the fully-built map.

Because the map is built off to the side and swapped in with a single atomic step
([Concurrency](concurrency.md)), a concurrent align never observes a partial or kd-tree-less map.

## C++ / Rust split

Rust owns the canonical map state, staging build, and atomic publication. C++ retains the ROS
pcd-loader service I/O and debug-map publication, then supplies loaded tiles through the
status-returning `node_map_update` boundary.

> Source: `src/scan_matcher.rs` (`apply_map_update`), `../src/node_map_update.rs`, `src/engine.rs`
> (`commit_from`, `clone_empty`), `../src/node_handle.rs` (`MapUpdateState`).
