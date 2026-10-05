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


@pytest.fixture(scope="session")
def tiny_hybrid_backbone(tmp_path_factory) -> Backbone:
    """Qwen3.5-style hybrid: 3 linear-attention (gated delta) layers + 1 full attention.
    Recurrent layers ignore attention masks, so this exercises the forked-branch mode."""
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    path = tmp_path_factory.mktemp("tiny-qwen35")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    Qwen3_5ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return Backbone(str(path))


@pytest.fixture(params=["attention", "hybrid"])
def any_backbone(request, tiny_backbone, tiny_hybrid_backbone) -> Backbone:
    return tiny_backbone if request.param == "attention" else tiny_hybrid_backbone


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
