"""The embedding models, behind one interface.

Each embedder turns audio windows and text queries into unit-length float32
vectors in the same space, so text-to-audio and audio-to-audio search are both
a dot product.
"""

import numpy as np
import torch


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _unit(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x.float(), dim=-1).cpu().numpy().astype(np.float32)


class ClapEmbedder:
    """LAION CLAP: 48 kHz input, trained on captioned sound libraries and music."""

    name = "clap"
    label = "CLAP"
    repo = "laion/larger_clap_general"
    sr = 48_000
    window_s = 10.0
    batch_size = 16

    def __init__(self) -> None:
        from transformers import ClapModel, ClapProcessor

        self.device = _device()
        self.model = ClapModel.from_pretrained(self.repo).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(self.repo)

    @torch.no_grad()
    def embed_audio(self, windows: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(audio=windows, sampling_rate=self.sr, return_tensors="pt")
        return _unit(self.model.get_audio_features(**inputs.to(self.device)).pooler_output)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return _unit(self.model.get_text_features(**inputs.to(self.device)).pooler_output)


class GemmaEmbedder:
    """EmbeddingGemma 2: 16 kHz input, long context, speech and environmental audio."""

    name = "gemma"
    label = "EmbeddingGemma 2"
    repo = "google/embeddinggemma-2"
    sr = 16_000
    window_s = 30.0
    batch_size = 4
    # Shorter clips are zero-padded: the audio encoder needs a minimum number of frames.
    min_samples = 16_000

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        # float32, not float16: the model card warns float16 silently returns
        # degraded or NaN embeddings. The vision encoder is left out.
        self.model = SentenceTransformer(
            self.repo,
            device=_device(),
            config_kwargs={"vision_config": None},
            model_kwargs={"torch_dtype": torch.float32},
        )

    def embed_audio(self, windows: list[np.ndarray]) -> np.ndarray:
        padded = [np.pad(w, (0, max(0, self.min_samples - len(w)))) for w in windows]
        items = [{"audio": {"array": w, "sampling_rate": self.sr}} for w in padded]
        return self._encode(items)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt_name="SearchQuery")

    def _encode(self, items, **kwargs) -> np.ndarray:
        vecs = self.model.encode(
            items, batch_size=self.batch_size, normalize_embeddings=True, **kwargs
        )
        return np.asarray(vecs, dtype=np.float32)


EMBEDDERS = {cls.name: cls for cls in (ClapEmbedder, GemmaEmbedder)}


def load(name: str):
    return EMBEDDERS[name]()
