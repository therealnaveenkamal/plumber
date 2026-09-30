"""Runner hooks: carry per-request decision metadata from ``SamplingParams.extra_args`` to the plumb model.

vLLM-internal: wraps four methods of the V2 model runner (vllm/v1/worker/gpu/model_runner.py, v0.30.0), each a no-op
unless the loaded model is a plumb model:

  add_requests     new requests: keep ``extra_args["plumb"]`` (suffix start, option spans, DECIDE, option count)
  finish_requests  finished / preempted requests: drop their state (a preempted request is re-added on resume)
  prepare_inputs   the batch is laid out: write the per-token adapter mask before the forward
  sample           the forward is done: keep suffix rows, run the head, and let compute_logits answer
"""

from __future__ import annotations


def _plumb(runner):
    m = getattr(runner, "model", None)
    for _ in range(4):  # unwrap cudagraph / compile wrappers until the model with the state
        if m is None:
            return None
        state = getattr(m, "plumb", None)
        if state is not None and hasattr(state, "on_batch"):
            return state
        m = getattr(m, "runnable", None) or getattr(m, "module", None) or getattr(m, "model", None)
    return None


def install() -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as R

    if getattr(R, "_plumb_hooked", False):
        return
    add, finish, prepare, sample = R.add_requests, R.finish_requests, R.prepare_inputs, R.sample

    def add_requests(self, scheduler_output):
        add(self, scheduler_output)
        p = _plumb(self)
        if p is None:
            return
        for req in scheduler_output.scheduled_new_reqs:
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            if "plumb" in extra:
                p.meta[req.req_id] = extra["plumb"]
                p.rows.pop(req.req_id, None)

    def finish_requests(self, scheduler_output):
        p = _plumb(self)
        if p is not None:
            gone = set(scheduler_output.finished_req_ids) | set(
                scheduler_output.preempted_req_ids or ()
            )
            for rid in gone:
                p.forget(rid)
        return finish(self, scheduler_output)

    def prepare_inputs(self, *args, **kwargs):
        batch = prepare(self, *args, **kwargs)
        p = _plumb(self)
        if p is not None:
            p.on_batch(batch)
        return batch

    def sample_(self, hidden_states, input_batch, grammar_output):
        p = _plumb(self)
        if p is not None:
            p.before_sample(input_batch)
        try:
            return sample(self, hidden_states, input_batch, grammar_output)
        finally:
            if p is not None:
                p.override.clear()

    R.add_requests, R.finish_requests, R.prepare_inputs, R.sample = (
        add_requests,
        finish_requests,
        prepare_inputs,
        sample_,
    )
    R._plumb_hooked = True
