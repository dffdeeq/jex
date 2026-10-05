import torch

from jex.backbone import Backbone
from jex.lora import answer_label_logits, train_lora
from jex.schema import parse_request


def test_lora_learns_and_round_trips(tiny_backbone, payload, tmp_path):
    path = tiny_backbone.name  # the tiny model directory from conftest
    bb = Backbone(path)
    records = [{"state": payload["state"], "questions": payload["questions"]}]
    req = parse_request(payload)
    # push every question towards its first option
    targets = [[torch.eye(q.num_options)[0] for q in req.questions]]

    def first_option_prob(model_bb):
        with torch.no_grad():
            logits = answer_label_logits(model_bb, [model_bb.encode(req)])[0]
        return [torch.softmax(lg, -1)[0].item() for lg in logits]

    before = first_option_prob(bb)
    peft_model = train_lora(bb, records * 8, targets * 8, epochs=3, lr=5e-3, batch_records=2, log=lambda *_: None)
    after = first_option_prob(bb)
    assert all(a > b for a, b in zip(after, before)), (before, after)

    peft_model.save_pretrained(tmp_path / "adapter")
    merged = Backbone(path, adapter=str(tmp_path / "adapter"))
    reloaded = first_option_prob(merged)
    assert max(abs(a - b) for a, b in zip(after, reloaded)) < 1e-4
    # the merged model serves through the normal prefill-only path
    feats = merged.run([merged.encode(req)])[0]
    assert abs(torch.softmax(feats[0].prior, -1)[0].item() - reloaded[0]) < 1e-4
