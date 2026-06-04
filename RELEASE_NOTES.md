#### 0.1.0 - June 2026
* Initial release: F# port of `fastcoref` running on TorchSharp.
* `CorefModel(modelDir, kind, ?device)` with `CorefKind.FCoref` and `CorefKind.LingMess`.
* `Predict` / `PredictBatch` returning `CorefResult` with `Cluster<Mention>` (character-level spans, UMX-tagged `CharIdx`).
* `LoadReport` exposing loaded/missing parameter counts after `state_dict` load.
* xUnit test suite gated on `FASTCOREF_MODELS_DIR`.
