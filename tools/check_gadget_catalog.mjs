// INV-509 acceptance: the client gadget catalog must mirror the server's
// trusted-renderer catalog (names, params, snapshots, predicates).
// Usage: node tools/check_gadget_catalog.mjs   (exit 0 = consistent)
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";

const client = JSON.parse(readFileSync(new URL("../client/js/gadget-catalog.json", import.meta.url), "utf8"));
const py = JSON.parse(execSync(
  ".venv/bin/python -c \"import json;from src.content.gadget_tasks import GADGET_CATALOG as c;print(json.dumps(c))\"",
  { cwd: new URL("..", import.meta.url).pathname }).toString());

const problems = [];
for (const gadget of new Set([...Object.keys(client), ...Object.keys(py)])) {
  const c = client[gadget], p = py[gadget];
  if (!c) { problems.push(`${gadget}: missing from client catalog`); continue; }
  if (!p) { problems.push(`${gadget}: missing from server catalog`); continue; }
  if (JSON.stringify(c.params) !== JSON.stringify(p.params)) problems.push(`${gadget}: params diverge`);
  if (JSON.stringify(c.snapshots) !== JSON.stringify(p.snapshots)) problems.push(`${gadget}: snapshots diverge`);
  if (JSON.stringify(c.predicates) !== JSON.stringify(p.predicates)) problems.push(`${gadget}: predicates diverge`);
}
if (problems.length) {
  console.error("catalog mismatch:\n" + problems.map((x) => ` - ${x}`).join("\n"));
  process.exit(1);
}
console.log(`gadget catalog consistent: ${Object.keys(py).length} trusted renderers`);
