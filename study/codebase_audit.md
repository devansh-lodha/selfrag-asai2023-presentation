# Codebase Audit: The Released Self-RAG Code vs the Paper

This file replaces the two earlier audits (`CODEBASE_AUDIT.md` at the project root and `study/codebase_audit_and_bugs.md`). Every item was re-checked against `reference/code/` by reading the source, and the line numbers below are current. Nothing was executed (this environment has no GPU, PyTorch, or vLLM), so runtime behavior is inferred from the code; items that depend on library defaults rather than on the repository's own code are marked "inferred". Issues in the paper itself are in `study/paper_audit.md`.

---

## 1. Summary

What the release provides: the main README links the 7B and 13B checkpoints and the 150k generator training data on Hugging Face, plus evaluation files with pre-retrieved passages, and the repository contains scripts for every stage (GPT-4 labeling, Critic training, generator data creation, generator training, inference, and evaluation).

The code implements the method the paper describes: $15$ added special tokens, a Critic distilled from GPT-4 labels, offline insertion of passages and reflection tokens, generator training with passage masking, and inference that ranks passage-conditioned candidates by critique scores. As committed, however, it does not reproduce the paper's pipeline cleanly:
- Several documented commands crash before finishing (Section 3.A).
- Some inference paths compute different scores than the paper describes (Section 3.B). The largest deviations: the short-form adaptive threshold is computed on raw log-probabilities, the long-form segment score reduces to $s(\text{IsRel}) + s(\text{IsSup})$, and long-form beam search leaks sibling text into stored prompts.
- Generator data creation contains bugs and heuristics that the paper does not describe (Section 3.C), including a loss-masking bug that removes the training signal for intermediate sentences in every multi-passage example.
- Several settings differ from the paper (Section 3.D).

How to use this in the presentation: the paper's numbers come from the authors' own runs, and this audit can neither confirm nor refute them. What it does support are statements about the release: some figures cannot be reproduced with the scripts as committed (Figure 3b), some mechanisms in the paper are absent from the released long-form code (the utility score, per-segment retrieval), and the training data has properties the paper does not describe. Frame these as "released code vs paper", not as evidence that the reported results are wrong.

---

## 2. Corrections to the Earlier Audits

- The multi-paragraph masking bug in `retrieval_lm/finetune.py` does not raise an `AssertionError`, as both earlier audits claimed. `labels` is a PyTorch tensor, and iterating over it yields views into the same storage, so the second `<paragraph>` already reads `-100` when the outer loop reaches it, the branch is skipped, and the `assert` never runs. The bug is silent over-masking (item C23).
- "`always_retrieve` mode was used for Table 2" is not supported by the repository. The README's ARC-Challenge and PubHealth commands pass no `--mode` (the default runs the inverted threshold branch, item B1), the QA command leaves the mode as a `MODE` placeholder, the ASQA command uses `always_retrieve`, and the FactScore command uses the default adaptive mode.
- "The `No retrieval` ablation for long-form could not have run" is beside the point: the paper reports no long-form no-retrieval result (the ASQA cells for it in Figure 3a are empty).
- The `run_short_form.py` fallback check is on line 83 and raises `UnboundLocalError` (a subclass of `NameError`), because `pred` is assigned later in the same function.
- The claim that a community team "replicated the core performance gains" was removed: the README only links a community-trained Mistral-7B variant in its FAQ.
- Line numbers in the earlier audits (and in many study module files) were off by up to about $90$ lines; the numbers below were re-verified.

---

## 3. Inventory

### 3.A Crashes on Documented or Paper-Relevant Paths

- A1. `retrieval_lm/run_short_form.py:315`: `generate()` passes `max_depth=args.max_depth` to `call_model_rerank_w_scores_batch` (lines 51-54), which has no such parameter and no `**kwargs`. Effect: `TypeError` on the first example of every short-form run in the README.
- A2. `retrieval_lm/run_short_form.py:83`: `do_retrieve = "[Retrieval]" in pred` runs before `pred` is assigned (the generated text is in `pred_text`, line 61). Effect: `UnboundLocalError` in adaptive mode without `--threshold`.
- A3. `retrieval_lm/run_short_form.py:178-180`: `len(results) == 1` is used to detect "no retrieval", but with `--mode always_retrieve --ndocs 1` there is also exactly one result, and `pred` is then the vLLM output object left over from the scoring loop. Effect: `postprocess_answer_option_conditioned(pred)` fails, so the "Retrieve top 1" setting cannot run through this path.
- A4. `retrieval_lm/run_long_form_static.py:159-165`: `--mode no_retrieval` returns `prediction_tree` before it is assigned. Effect: `UnboundLocalError`.
- A5. `retrieval_lm/run_long_form_static.py:192-198` and `375-380`: when the root decides not to retrieve, `call_model_beam_batch` returns a plain string and an empty tree; the FactScore branch then takes `result[0]` (the first character of the answer, line 378) and indexes `intermediate["original_splitted_sentences"]` (line 380). Effect: `KeyError`. The README's FactScore command uses the default adaptive mode, so any biography prompt without retrieval stops the run (the ASQA branch guards this on line 427).
- A6. `data_creation/train_special_tokens.py:27`: `from ..retrieval_lm.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn` is a relative import, but the README runs the file as a script (`cd data_creation` then `torchrun ... train_special_tokens.py`). Effect: `ImportError: attempted relative import with no known parent package` before training starts.
- A7. `data_creation/train_special_tokens.py:236`: reads `PROMPT_DICT["prompt_no_input_paragraph"]` and `PROMPT_DICT["prompt_no_input_separated"]`, which `PROMPT_DICT` (lines 34-41) does not define. Effect: `KeyError` when building the dataset.
- A8. `retrieval_lm/finetune.py:449-454`: `special_token_dict` is created only under `if args.use_special_tokens is True:` but written unconditionally on line 451. Effect: `NameError` when training without reflection tokens (as a `Llama2-FT`-style baseline would need).
- A9. `retrieval_lm/finetune.py:518-522`: debug prints of fixed indices up to `train_dataset[10000]`. Effect: `IndexError` for training sets with at most $10,000$ examples, such as the $5\text{k}$ and $10\text{k}$ data-scaling runs.
- A10. `retrieval_lm/run_baseline_lm.py:46`: the exception handler of `call_model_instructgpt` assigns `results` instead of `result`. Effect: `UnboundLocalError` on any OpenAI API error.
- A11. `data_creation/critic/gpt4_reward/chatgpt_relevance.py:93`: `PROMPT_DICT["context"]` does not exist. Effect: `KeyError` unless `--multi_retrieval` is passed.
- A12. `retrieval_lm/src/beir_utils.py:192`: `GenericDataLoader(data_folder=data_folder)` uses an undefined name in the `cqadupstack` branch. Effect: `NameError`.
- A13. `data_creation/generator/run_reward_vllm.py:282` and `243-244`: the remainder block reads `idx` from the main loop, which never runs when a file has fewer than `batch_size` ($10$) items, and the resume option expects a dict with `"preds"` although the script saves a list. Effect: `NameError` on tiny inputs; `TypeError` when resuming.

### 3.B Inference That Differs from the Paper

- B1. `retrieval_lm/run_short_form.py:75-81`: the adaptive-retrieval ratio is computed on raw log-probabilities (`score_dict[tok] = float(prob)`, no `np.exp`). Because log-probabilities are negative, the ratio is inverted: a model confident that it needs retrieval skips it, and vice versa. The README's ARC-Challenge and PubHealth commands pass `--threshold 0.2` and no `--mode`; the argparse default is the string `"default"` (line 287), which falls into this branch. Effect: the documented closed-set commands run inverted adaptive retrieval. Which mode produced Table 2 cannot be determined from the repository.
- B2. `retrieval_lm/run_short_form.py:285`: `--w_use` defaults to `1.0`, while the paper's default is $0.5$; the README's short-form commands do not override it.
- B3. `retrieval_lm/run_long_form_static.py:68`: the utility block loops over `grd_tokens` ($3$ tokens) instead of `ut_tokens` ($5$), so the check on line 82 never passes. Effect: `utility_score` is always $0.0$ in long-form decoding.
- B4. `retrieval_lm/run_long_form_static.py:221-224`: `use_seqscore` is not forwarded to `run_step_generation_batch` (default `False`, line 13). Effect: long-form ranking never includes $\exp(\text{seq\_score})$, even though the README passes `--use_seqscore`.
- B5. `retrieval_lm/run_long_form_static.py:364`: `generate()` hardcodes `w_rel=1.0, w_sup=1.0, w_use=0.5`, ignoring `--w_rel`, `--w_sup`, `--w_use` (lines 329-334). Effect: together with B3 and B4, every long-form run ranks segments by exactly $1.0 \cdot s(\text{IsRel}) + 1.0 \cdot s(\text{IsSup})$, and Figure 3b (which varies $w_{\text{sup}}$) cannot be reproduced without editing the script.
- B6. `retrieval_lm/run_long_form_static.py:217-243`: `prev_generation` holds the parent's text but is overwritten inside the child loop, so each child after the first stores a prompt (line 234) containing the previous sibling's segment. Effect: when such a child survives pruning, its next-depth expansions are conditioned on a sentence from a different branch that never appears in the output.
- B7. `retrieval_lm/run_long_form_static.py:171-191`: in adaptive mode the root retrieves only if the greedy text of up to $25$ tokens already contains `"[Retrieval]"` and the normalized probability at position $0$ exceeds the threshold; with no `--threshold` it never retrieves. Effect: stricter than Appendix A.3, where the probability test alone can trigger retrieval; applies to the README's FactScore command.
- B8. `retrieval_lm/run_long_form_static.py:118-137`: the mid-generation remap tests $(p(\text{[Retrieval]}) + p(\text{[Continue]})) / (p(\text{[Retrieval]}) + p(\text{[No Retrieval]})) > \delta$, a rule absent from the paper whose ratio can exceed $1$; line 128 compares the resulting boolean against the threshold again (correct only for $\delta < 1$); line 137 appends a stray `"[No Retrieval]"` after the last substring (harmless).
- B9. `retrieval_lm/run_long_form_static.py:41-46`: a relevance token missing from the returned log-probabilities is stored as the probability `-100` (instead of $\exp(-100) \approx 0$). Effect: a meaningless relevance score in that rare case (`logprobs=32000` of $32016$ tokens).
- B10. `retrieval_lm/run_long_form_static.py:404-424`: ASQA citations write the path position `[idx]`, but `docs` grows only for kept sentences, so any skipped (empty or duplicate) sentence misaligns every later citation; `".[Continue to Use Evidence]"` is replaced by `" [1]. "`, which always cites the first document.
- B11. `retrieval_lm/run_long_form_static.py` (whole file): the script is "static": it receives one fixed list of $K$ pre-retrieved passages per question and reuses it at every depth, never calling the retriever again. Algorithm 1 retrieves per segment with the current context. The README states that only this pre-retrieved version is released.
- B12. `retrieval_lm/run_long_form_static.py:274`: the bottom-up traversal increments `current_level` instead of decrementing; the loop ends only because the root's parent is `None`.
- B13. `retrieval_lm/run_short_form.py:367`: the printed retrieval frequency divides by `len(final_results)`, which is the number of keys ($8$), not the number of examples.
- B14. `retrieval_lm/run_baseline_lm.py:175-177`: for full batches, the closed-set normalization `postprocess_answers_closed(...)` is immediately overwritten by `item["output"] = pred` (the remainder batch at line 199 keeps it). Effect: baseline closed-set scoring relies on substring matching of raw outputs.
- B15 (inferred). `run_short_form.py` (lines 57-58, 88-89), `run_long_form_static.py` (lines 20-21, 172-173), and `data_creation/generator/run_reward_vllm.py` (lines 131-132) never pass `skip_special_tokens=False`. With `vllm==0.2.6` (pinned in `requirements.txt`), `SamplingParams` defaults to `skip_special_tokens=True`, which removes registered special tokens from the decoded text; the README's Quick start passes the flag. Effect: text-based checks break (long-form `"[Retrieval]"` expansion and remapping, the long-form root decision, and the Critic label parsing used for data creation). Short-form scoring reads token ids and is mostly unaffected.
- B16. `retrieval_lm/utils.py:78-81`: `fix_spacing` inserts a space after any period between word characters, so `"3.14"` becomes `"3. 14"` and `"U.S."` becomes `"U. S."` in long-form outputs before evaluation.
- B17. `retrieval_lm/run_short_form.py:331`: `pred[0]` is read without checking for an empty prediction. Effect: `IndexError` on an empty output.

### 3.C Training-Data Construction (Critic and Generator)

- C1. `data_creation/process_data/tulu.py:38`: `if dataset_name == "sharegpt" or "oasst1":` is always true, so language detection runs on all five Open-Instruct subsets; when detection raises an exception, the item is kept.
- C2. `data_creation/critic/gpt4_reward/chatgpt_need_retrieval.py:128-146`: the three-way templates are swapped, so sentences that do have preceding sentences are sent to GPT-4 with the template that omits them (and only sentences with `sent_idx > 0` are kept, line 197). Effect: most three-way retrieval labels were collected without the preceding context that the paper's Table 9 prompt includes.
- C3. `chatgpt_groundness.py:101` and `chatgpt_relevance.py:80`: the score parser searches only for digits, so bracketed labels such as `[Fully supported]` give `score = ""`; `combine_chat_gpt_reward.py` recovers them from `raw_output` (lines 98-99, 162-163, 187-188). In `chatgpt_utility.py:53` the same loop keeps the largest digit present in the score line.
- C4. `chatgpt_groundness.py:205`, `chatgpt_relevance.py:159`, `chatgpt_utility.py:104`: `"explanation": score` is saved instead of the explanation (it survives in `raw_output`).
- C5. `data_creation/critic/gpt4_reward/README.md` (line 3) says "We use GPT-4 by default", but the scripts default to `gpt-3.5-turbo` (for example, `chatgpt_need_retrieval.py:153`), so GPT-4 must be requested explicitly.
- C6. `combine_chat_gpt_reward.py:57-67`: the "without preceding" three-way template is identical to the "with preceding" one; line 126 tags multi-sentence items that have preceding sentences as task `"retrieval"` instead of `"multi_retrieval"` (this only affects task-filtered Critic evaluation); line 194 keeps $70\%$ of `[Relevant]` examples, a rebalancing the paper does not mention.
- C7. `data_creation/generator/run_reward_vllm.py:153-169` and `171-189`: `process_data` is defined twice; its `_no_preceding` lookup (lines 181-183) is unreachable because `multi_retrieval` is never passed. Effect: first-sentence `IsSup` prompts contain the literal line `"Preceding sentences: None"`, while the Critic was trained on `ground_multi_input_wo_preceding`, which omits it.
- C8. `data_creation/generator/run_reward_vllm.py:281-299`: the remainder batch is never appended to `predicted_results`. Effect: the last `len % 10` predictions of every file are missing from the saved output, which removes candidate passages (or whole sentences) downstream.
- C9. `data_creation/generator/run_reward_vllm.py:269-276`, `302`: the Critic-accuracy path counts a prediction as correct when either string contains the other and excludes empty predictions from the mean. The paper does not say whether Figure 5 used this path.
- C10. `data_creation/generator/README.md` (Step 4b command): runs the multi-sentence Critic with the two-way `retrieval_multi_instruction` / `retrieval_multi_input` prompt, while the Critic was trained only on the three-way prompt (`combine_chat_gpt_reward.py:124-126`). Obtaining `[Continue to Use Evidence]` labels requires the three-way prompt.
- C11. `data_creation/generator/create_retrieval_data.py:74`: `retrieval_necessity[q_id] is False` never holds because predictions are stored as text. Effect: every example goes through sentence-level retrieval, although the generator README says only examples that need retrieval do; the initial decision is applied later in `postprocess_data.py`.
- C12. `data_creation/generator/create_retrieval_data.py:111-112`: the whole-output pass strips the delimiter `"\n\nInput: "` (only for `nq`, `fever`, `wow`), but the preprocessors write `"## Input:\n\n"`. Effect: Pass 1 queries keep the task instruction; Pass 1 only supplies the `evidence` passage for the multi-sentence Critic.
- C13. `data_creation/generator/create_retrieval_data.py:103-104`: one `splitted_data` entry is appended per non-skipped sentence (duplicates; `postprocess_data.py` reads only the first).
- C14. `data_creation/generator/create_prompt_data.py:72`: integer batch splitting drops the last `len % num_jobs` items.
- C15. `data_creation/generator/postprocess_data.py:293, 302, 307, 310`: `random.sample(range(len(irrelevant_p_indices)), k=1)[0]` returns a position in the list, not a passage index. Effect: the "irrelevant" pick is really a random choice among the top-ranked passages, so the data has fewer `[Irrelevant]` examples than intended.
- C16. `data_creation/generator/postprocess_data.py:309-310`: for `nq` and `fever`, whenever any irrelevant passage exists, the choice is overwritten after the cascade, even a `[Fully supported]` one; this also makes the `nq` rule on lines 292-293 redundant.
- C17. `data_creation/generator/postprocess_data.py:246-248, 257-259, 299-300, 304-306, 312-317, 328-329`: `continue` inside the sentence loop skips appending the sentence. Effect: some examples lose sentences of their original answer.
- C18. `data_creation/generator/postprocess_data.py:222`: the no-retrieval branch compares the raw Critic string, not the normalized token, so a raw `"No"` takes the retrieval branch.
- C19. `data_creation/generator/postprocess_data.py` heuristics absent from the paper: keep $40\%$ of `[No Retrieval]` examples (line 224); drop $50\%$ of `nq` outputs containing any `[No Retrieval]` (lines 351-352); rewrite `[Utility:1]` to `[Utility:5]` for `true`/`false` outputs (lines 211-212); relabel `[Relevant]` + `[No support / Contradictory]` as `[Irrelevant]` for `nq`, `fever`, `wow` (lines 325-326); drop outputs that start with `[Continue to Use Evidence]` (line 368).
- C20. `data_creation/generator/postprocess_data.py:14-18`: `ground_tokens_names` and `utility_tokens_names` hold each other's lists (harmless; only concatenated on line 23).
- C21. `data_creation/process_data/arc.py:32-36`: choice labels are overwritten with A-D and only questions with fewer than four options are filtered, so a question with more options would keep an answer key that is not among the four shown.
- C22. `retrieval_lm/src/finetuning_data.py:28`: `normalize_text.normalize if normalize_text else ...` tests the imported module (always truthy), so retriever fine-tuning data is always normalized.
- C23. `retrieval_lm/finetune.py:269-285` (same logic in `data_creation/train_special_tokens.py:212-224`): the inner search for `</paragraph>` has no `break`, so `end_idx` becomes the last `</paragraph>` in the sequence and everything from the first `<paragraph>` to it is masked. Effect: in every multi-passage generator example, the intermediate sentences and their reflection tokens get no loss, and only the text after the last passage is trained; this happens silently (see Section 2). Critic targets contain no paragraphs, so only the generator is affected.
- C24. Combined effect: because of C17 and C19, the final generator data is smaller than the set of examples entering `postprocess_data.py`. The paper's Table 3 ($145,619$, captioned as generator training data statistics) does not say which stage it counts.

### 3.D Settings and Documentation That Differ from the Paper

- D1. `retrieval_lm/script_finetune_13b.sh:22`: `--max_seq_length 1536`; Appendix B.1 says $1,524$.
- D2. README Critic training command: $2$ GPUs $\times$ batch $1$ $\times$ accumulation $8$ $= 16$, warmup $1\%$, cosine schedule; Appendix B.1 says all models used batch $128$, $3\%$ warmup, and linear decay (epochs and learning rate match).
- D3. README (line 230): "8 A100 with 40 GRAM for 7B" and "4 a100 with 80 GB" for 13B; Appendix B.1 says $4$ A100 80GB; both training scripts set $4$ GPUs. The README also refers to a `training_13b` script, while the file is `script_finetune_13b.sh`.
- D4. README FactScore command: `--max_new_tokens 300` per segment; Appendix B.2 says $200$ per segment for Self-RAG.
- D5. Short-form `w_use` default (B2): $1.0$ in code vs $0.5$ in the paper.
- D6. `retrieval_lm/utils.py:34-41`: `TASK_INST` lacks the formatting sentences that the paper's Table 5 shows for PubHealth and ARC-Challenge (Appendix B.2 says instructions were adjusted per model). Inference prompts also join instruction and question differently from training: `"\n\n### Input:\n"` for ARC-Challenge and `"\n\n## Input:\n\n"` for PubHealth (`run_short_form.py:241` and `244`) versus `"## Input:\n\n"` in the training data.
- D7. The Critic initializes new token embeddings to the mean of existing rows (`train_special_tokens.py:143-164`), while the generator (`finetune.py:474-476`) uses Hugging Face's default initialization.
- D8. The pad token is spelled `[PAD]` in `train_special_tokens.py:30` and `<pad>` in `finetune.py:454`; both scripts add it after the $15$ reflection tokens, so it has id $32015$ in both.
- D9. Retrieval threshold normalization: Section 3.3 says "over all output tokens in Retrieve", Appendix A.3 uses Yes and No only; the short-form code uses Yes/No (with B1), and the long-form code uses the stricter rule of B7.
- D10. `requirements.txt` pins `vllm==0.2.6`, while the README asks for "the latest version of vllm" so that `skip_special_tokens` can be set (see B15).

### 3.E Minor Issues and Dead Code

- E1. `retrieval_lm/finetune.py:241` prints the token length of every tokenized string; lines 518-528 print every training example and dump the whole dataset to `processed.json` before training.
- E2. `retrieval_lm/finetune.py:545`: `no_decay = ["bias", "layer_norm.weight"]` matches no Llama-2 parameter (no biases; norms are named `input_layernorm`, `post_attention_layernorm`, `norm`), so the no-decay group is empty; harmless only because the scripts set weight decay to $0$.
- E3. `data_creation/critic/gpt4_reward/chatgpt_inference.py:50`: the model name is hardcoded to `gpt-3.5-turbo`.
- E4. `retrieval_lm/utils.py:52` (`control_tokens`), `utils.py:85` (`postprocess`), and `postprocess_data.py:14`: `[Continue to Use Evidence]` is missing from the strip lists (worked around for ASQA by B10's replacement).
- E5. `retrieval_lm/run_short_form.py:165`: the saved `"ut_score_dict"` field holds the utility score, not the dictionary.
- E6. `data_creation/generator/postprocess_data.py:255`: `need_retrieval_decision_token == "[Retrieval]"` is a comparison with no effect.
- E7. `data_creation/generator/README.md`: several commands miss line-continuation backslashes (for example, after `--inst_mode retrieval_instruction` and `--input_mode ground_multi_input`), so they fail if pasted as written.
- E8. `run_short_form.py:287` and `run_long_form_static.py:337`: `--mode` defaults to `"default"`, which is not one of its `choices` (argparse does not validate defaults); the scripts treat it as adaptive retrieval.

---

## 4. Impact on the Paper's Claims

- Consistent with the paper: the token vocabulary and ids ($32000$-$32015$), the Critic objective, the generator objective with passage masking (for single-passage examples), and the short-form critique-score formulas (Appendix A.3).
- Long-form results (ASQA, FactScore): as released, decoding uses fixed weights, no utility score, and no sequence score (B3-B5), reuses a fixed passage set (B11), leaks sibling text into deeper expansions (B6), and can misnumber citations (B10). The paper describes a more general procedure.
- Figure 3b ($w_{\text{sup}}$ sweep): not reproducible with the released long-form script without edits (B5).
- Short-form adaptive retrieval and Figure 3c: the documented closed-set commands run the inverted threshold (B1); the paper's figure must come from a different code path or version.
- Data-scaling runs (Figure 4): the $5\text{k}$ and $10\text{k}$ runs crash in `finetune.py` as committed (A9).
- Generator training data: if the released `finetune.py` produced the released checkpoints, intermediate sentences and their reflection tokens in multi-passage examples received no loss (C23); the data also reflects undocumented downsampling, relabeling, and dropped sentences (C15-C19).
- Not determinable from the repository: which code version produced the paper's numbers.
