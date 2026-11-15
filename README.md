# Free model aggregator 02

One server that serves the OpenAI Chat Completions API and the Anthropic Messages
API over the fourteen free OpenRouter models, in JSON and in server-sent events.
Each prompt is served by the best-rated free model that can answer it, with a
buffered token check before the answer is relayed and a distinct-model failover
on refusal. Six prompts, six calls, no retries of the same model.

Node 22, standard library only. `node main.ts --list` prints the catalog.
