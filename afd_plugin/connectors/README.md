# Connector Package Layout

AFD connector implementations are grouped by backend:

- `gpu/`: GPU-only connector implementations. `P2pNcclAFDConnector` is implemented by
  `afd_plugin.connectors.gpu.p2p`.
- `npu/`: NPU-only connector implementations. `CAMP2pAFDConnector` is implemented
  by `afd_plugin.connectors.npu.camp2p`, and `CAMAsyncAFDConnector` is implemented
  by `afd_plugin.connectors.npu.async_cam`.

The vLLM 0.26 support matrix validates GPU `P2pNcclAFDConnector` and synchronous
NPU `CAMP2pAFDConnector`. `CAMAsyncAFDConnector` remains experimental and was
not revalidated during the v0.26 upgrade; its PCP8 recipe is a historical
v0.19.1 experiment.

Shared connector contracts, metadata containers, factory registration, and
backend-neutral helpers stay in `afd_plugin.connectors`.
