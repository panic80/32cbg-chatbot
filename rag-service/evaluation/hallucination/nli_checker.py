"""NLI-based factual consistency checker."""

from typing import Dict, List, Optional, Tuple


class NLIChecker:
    """NLI-based factual consistency checker using DeBERTa or similar."""

    # Standard label names we work with
    STANDARD_LABELS = {"entailment", "neutral", "contradiction"}

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        """Initialize NLI checker.

        Args:
            model_name: HuggingFace model name for NLI
            device: Device to run on (cpu, cuda, mps)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._label_map: Dict[int, str] = {}

    def _load_model(self) -> bool:
        """Load NLI model and tokenizer lazily.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import torch

            print(f"Loading NLI model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Extract label mapping from model config
            if hasattr(self.model.config, "id2label"):
                raw_map = self.model.config.id2label
                # Normalize label names to lowercase
                self._label_map = {
                    int(k): v.lower() for k, v in raw_map.items()
                }
                print(f"Label mapping: {self._label_map}")
            else:
                # Fallback to common mapping
                self._label_map = {
                    0: "contradiction",
                    1: "entailment",
                    2: "neutral",
                }

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            else:
                self.device = "cpu"

            self.model.eval()
            self._loaded = True
            print(f"NLI model loaded on {self.device}")
            return True

        except ImportError as e:
            print(f"Warning: Could not load NLI model - missing dependencies: {e}")
            print("Install with: pip install transformers torch")
            return False
        except Exception as e:
            print(f"Warning: Could not load NLI model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if NLI checking is available."""
        return self._load_model()

    def check_claim_against_source(
        self,
        claim: str,
        source_text: str,
    ) -> Tuple[str, float]:
        """Check if claim is entailed by source.

        Args:
            claim: The claim to verify (hypothesis)
            source_text: The source text (premise)

        Returns:
            Tuple of (label, confidence) where label is
            'entailment', 'neutral', or 'contradiction'
        """
        if not self._load_model():
            return "neutral", 0.5

        import torch

        # Truncate source if too long
        if len(source_text) > 4000:
            source_text = source_text[:4000]

        # Tokenize: premise (source) -> hypothesis (claim)
        inputs = self.tokenizer(
            source_text,
            claim,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_label_idx = probs.argmax().item()
            confidence = probs[0][pred_label_idx].item()

        label = self._label_map.get(pred_label_idx, "neutral")
        return label, confidence

    def check_claims_batch(
        self,
        claims: List[str],
        sources: List[str],
    ) -> List[Tuple[str, float]]:
        """Batch check multiple claims against concatenated sources.

        Args:
            claims: List of claims to verify
            sources: List of source texts to check against

        Returns:
            List of (label, confidence) tuples
        """
        if not self._load_model():
            return [("neutral", 0.5) for _ in claims]

        if not claims:
            return []

        # Concatenate sources
        source_text = " ".join(sources)
        if len(source_text) > 8000:
            source_text = source_text[:8000]

        results = []

        # Process in batches
        for i in range(0, len(claims), self.batch_size):
            batch_claims = claims[i : i + self.batch_size]
            batch_results = self._check_batch(batch_claims, source_text)
            results.extend(batch_results)

        return results

    def _check_batch(
        self,
        claims: List[str],
        source_text: str,
    ) -> List[Tuple[str, float]]:
        """Check a batch of claims against source.

        Args:
            claims: List of claims
            source_text: Source text

        Returns:
            List of (label, confidence) tuples
        """
        import torch

        # Tokenize all pairs
        inputs = self.tokenizer(
            [source_text] * len(claims),
            claims,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(claims)):
            pred_label_idx = probs[i].argmax().item()
            confidence = probs[i][pred_label_idx].item()
            label = self._label_map.get(pred_label_idx, "neutral")
            results.append((label, confidence))

        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._loaded,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
