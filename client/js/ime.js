/**
 * Enter while a CJK IME candidate list is open confirms the candidate.
 * It must not submit the field.
 *
 * keyCode 229 is the legacy composition key. Some engines fire the confirming
 * Enter after compositionend with isComposing already false; armImeGuard keeps
 * a flag until the next turn so that Enter is still ignored.
 */
export function imeBlocksSubmit(event) {
  const target = event.target;
  return !!(event.isComposing || event.keyCode === 229 || (target && target.dataset && target.dataset.ime === "1"));
}

export function armImeGuard(el) {
  el.addEventListener("compositionstart", () => { el.dataset.ime = "1"; });
  el.addEventListener("compositionend", () => {
    setTimeout(() => { delete el.dataset.ime; }, 0);
  });
}
