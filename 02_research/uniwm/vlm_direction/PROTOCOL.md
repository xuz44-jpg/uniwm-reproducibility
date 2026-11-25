# VLM yaw-direction probe (fixed before running)

Question: can a frozen ~8B VLM tell the direction of a small pure-yaw camera rotation from two renders
taken at the same camera center, where a closed-form geometric reference is exact?

- Model: Qwen3-VL-8B-Instruct (8.8B, bf16, frozen, no fine-tuning), transformers, greedy decoding, max 8 new tokens.
  Weights stay bf16; layers beyond 12 GiB of GPU memory are offloaded to CPU by accelerate (shared GPU), which changes speed, not the model.
- Images: the existing Habitat references, 640x480, 12 positions (apartment_1 p00-p05, skokloster_castle p00-p05).
  Yaw is left-positive (checked: content shifts right for positive yaw, about 5.5 px per degree).
- Queries: 12 positions x 10 angles (+-3, 5, 7, 10, 15 deg) x 2 orders = 240 scored.
  fwd: Image 1 = yaw 0, Image 2 = yaw a; answer LEFT if a > 0.
  rev: images swapped; answer flips.
  Plus 12 identical pairs (yaw 0 twice), unscored, to read the default answer.
- Prompt: the single fixed string in queries.json; no prompt tuning after seeing answers.
- Primary metric: greedy accuracy per magnitude (n = 48 each) and overall (n = 240); unparsed answers count as wrong;
  exact two-sided binomial test against 50%.
- Secondary: first-token logit comparison LEFT vs RIGHT; LEFT rate; swap consistency (answer flips under swap);
  accuracy by scene and order.
- Reference: global phase correlation on the same pairs gets the sign right for 118/120 angle-position pairs.
- Any result is reported as is.

## Post hoc control (added after seeing the first run)

The first run answered RIGHT for 86% of scored queries. The prompt lists LEFT before RIGHT, so a preference
for the last option could cause this. The control reruns the same 252 queries with only the option order
reversed ("to the RIGHT or to the LEFT", "one word: RIGHT or LEFT."): run_qwen.py --options-reversed,
results_options_reversed.jsonl, summary_options_reversed.json.

## Post hoc variant: abstention options

Same 252 queries; the prompt says the camera "may have rotated" and offers LEFT, RIGHT, NONE (did not rotate)
and UNSURE (cannot tell); identical pairs are scored with NONE as the correct answer. Run in two option orders
(prompt_4way.txt, prompt_4way_rev.txt) with --labels LEFT,RIGHT,NONE,UNSURE; summarized by summarize_4way.py.
