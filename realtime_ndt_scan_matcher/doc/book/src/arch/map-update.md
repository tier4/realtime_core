# Map update

How the target map is refreshed at runtime without disturbing concurrent alignment.

## Policy vs mechanism

Deciding *when* an update is needed — first update, and whenever the vehicle has moved far enough
that the loaded map can no longer cover the LiDAR range — and whether a full rebuild is required is a
policy the consuming application owns. This crate provides the *mechanism*: the loaded cell ids live
in the engine, and `scan_matcher::apply_map_update` performs the staged build and atomic publish.

## The atomic commit

`scan_matcher::apply_map_update` performs:

1. `source.load(center, radius).await` — fetch the delta through the async `MapSource`
   port (empty delta ⇒ no-op, no republish).
2. Build the new map on a **private staging engine** — `engine.clone()` for an incremental update, or
   `engine.clone_empty()` for a rebuild — apply the added/removed tiles, and `create_kdtree()`.
3. `engine.commit_from(&staging)` — one atomic store publishes the fully-built map.

Because the map is built off to the side and swapped in with a single atomic step
([Concurrency](concurrency.md)), a concurrent align never observes a partial or kd-tree-less map.

## Division of responsibility

This crate owns the canonical map state, the staging build, and the atomic publication. Map I/O —
the pcd-loader service call and debug-map publication — belongs to the caller, which supplies loaded
tiles through the async `MapSource` port and drives `apply_map_update`.

> Source: `src/scan_matcher.rs` (`apply_map_update`), `src/engine.rs` (`commit_from`, `clone_empty`).
