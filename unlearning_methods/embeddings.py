# Text embeddings of the edit prompts, from the CLIP text encoder that conditions TRELLIS.
import torch

CONCEPT_ENCODER_MODEL = "openai/clip-vit-large-patch14"


class ConceptEncoder:
    def __init__(self):
        from transformers import AutoTokenizer, CLIPTextModel

        self.tokenizer = AutoTokenizer.from_pretrained(CONCEPT_ENCODER_MODEL)
        self.model = CLIPTextModel.from_pretrained(CONCEPT_ENCODER_MODEL).to(device="cpu", dtype=torch.float32)
        self.model.eval()

    @torch.no_grad()
    def concept_vectors(self, texts: list[str]) -> torch.Tensor:
        # One vector per prompt: the hidden state at its last content token (the one before EOS).
        enc = self.tokenizer(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        hidden = self.model(input_ids=enc["input_ids"]).last_hidden_state
        last_idx = enc["attention_mask"].sum(dim=1) - 2
        return hidden[torch.arange(hidden.shape[0]), last_idx]

    @torch.no_grad()
    def concept_sequences(self, texts: list[str], batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        # All 77 token states [n, 77, 768] and the attention mask [n, 77]. TRELLIS feeds every
        # position, padding included, to its cross-attention layers.
        hiddens, masks = [], []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[start : start + batch_size], max_length=77, padding="max_length",
                                 truncation=True, return_tensors="pt")
            hiddens.append(self.model(input_ids=enc["input_ids"]).last_hidden_state)
            masks.append(enc["attention_mask"])
        return torch.cat(hiddens, dim=0), torch.cat(masks, dim=0)
