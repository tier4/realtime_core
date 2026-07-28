# The localization utility crate

`realtime_localization_util` is the Rust counterpart of the C++ `autoware_localization_util`
shared library: ROS-free, `no_std` + `alloc`, and reused across the localization ports. It holds
three modules, re-exported by the engine crate under their original paths:

- `pose_buffer` (`realtime_localization_util/src/pose_buffer.rs`) — the `SmartPoseBuffer` port: a
  time-ordered buffer of stamped poses-with-covariance with twist-based linear interpolation (see
  [Divergences from upstream](../port/divergences.md) for its tf2 RPY conventions).
- `transform` (`realtime_localization_util/src/transform.rs`) — SE3 transforms, the NDT Gaussian
  fitting constants, and euler↔matrix conversions shared by the engine kernels.
- `tpe` (`realtime_localization_util/src/tpe.rs`) — the Tree-Structured Parzen Estimator behind
  the align-service pose search, covered in the next chapter.
