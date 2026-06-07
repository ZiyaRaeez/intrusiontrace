from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    use_pca: bool = True
    use_smote: bool = True
    use_kmeans: bool = True
    use_ensemble: bool = True


ABLATION_CONFIGS = (
    PipelineConfig(name="Full Model"),
    PipelineConfig(name="Without PCA", use_pca=False),
    PipelineConfig(name="Without SMOTE", use_smote=False),
    PipelineConfig(name="Without K-Means", use_kmeans=False),
    PipelineConfig(name="Without Ensemble", use_ensemble=False),
)
