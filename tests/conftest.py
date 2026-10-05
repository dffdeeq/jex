import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from jex.backbone import Backbone

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def tiny_backbone(tmp_path_factory) -> Backbone:
    """A randomly initialised 2-layer Qwen2 with the real tokenizer: the
    isolation / caching properties hold for any weights, so tests stay fast."""
    path = tmp_path_factory.mktemp("tiny-qwen")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    torch.set_num_threads(2)
    return Backbone(str(path))


@pytest.fixture
def payload():
    return {
        "state": {
            "from": "user@acme.com",
            "subject": "Duplicate charge on invoice #4411",
            "body": "Hi, we were billed twice for March. Please refund or we will cancel.",
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "upgrades"},
            },
            "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
            },
        },
    }
