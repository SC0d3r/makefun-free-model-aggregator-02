// free-model-aggregator-02's starter kit: one server, two client wires, and a real free model behind
// every answer.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth a fraction of the score: use it to see the shape of a trial, then
// replace its routing, its failover and its catalog with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's standard
//   library alone (nothing is installed; TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the key you pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its weight
//       (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a unique token
//       the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This starter sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the published ratings. It takes the first free model in the brief, in
//     the brief's order, that can meet the request's requirements. TODO(quality): rank the free
//     candidates by rating_milli at the prompt's requirements and prefer the best.
//   - failover (10%): a gateway call answered with a 4xx or a 5xx becomes a 502 to the client, and that
//     prompt is lost. TODO(failover): retry the prompt on the next free model that meets it, and count
//     every call it costs, because the tie-break is the number of gateway calls.
//   - rate limits: after a 429 it keeps picking the same first model, so one rate-limited model takes
//     every later prompt down with it. TODO(cooldowns): track each model's refusals and back off per
//     model, and never spend a call on a model that is cooling down.
//   - the catalog (5%): it lists every free model the brief names, which is exactly the fourteen ids
//     this column scores. TODO(catalog): keep it exact — register that same set and answer GET /v1/models
//     with it on both auth headers; the column checks the register body and both answers against exactly
//     those fourteen ids, so listing fewer loses it as surely as listing more.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The one free model that serves a request. TODO(quality): rank every free candidate that meets the
 * requirements by the brief's rating_milli (quality is 60% of the score) and prefer the best. This
 * baseline takes the first free model in the brief's order that meets the requirements.
 */
function routeFor(requires: Requirement): ModelRow | null {
  const routes = routesFor(requires);
  return routes.length === 0 ? null : routes[0];
}

/**
 * Every free model that meets the requirements, the brief's best rating first: the quality column is
 * 60% of the score and credits the rating of the model that really served the prompt, so the best model
 * that can serve it is the one to dial. Ties keep the brief's order, so the choice is deterministic.
 */
function routesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/**
 * The models one prompt may be dialled on, in order: the four best that meet it, each tried once.
 * A refusal, a broken body or an answer that does not carry the prompt's token moves to the next
 * model: retrying the same id spends a request a fresh model would have served, and only a
 * completion that uses tokens counts as a call in the tie-break.
 */
function attemptPlan(routes: ModelRow[]): ModelRow[] {
  return routes.slice(0, 4);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, text: string): void {
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
  } else {
    sendError(res, wire, 502, 'upstream_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
  }
}

/** The prompt's token, as the request body spells it. The answer must carry it back to count. */
function tokenOf(body: Json): string | null {
  const texts: string[] = [];
  const collect = (value: unknown): void => {
    if (typeof value === 'string') {
      texts.push(value);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) collect(item);
      return;
    }
    if (isRecord(value)) {
      for (const item of Object.values(value)) collect(item);
    }
  };
  collect(body.messages);
  for (const text of texts) {
    const match = /[A-Z][A-Z0-9]{2,}-\d{3,}/.exec(text);
    if (match !== null) return match[0];
  }
  return null;
}

/** The text of a message's content, whether it is one string or a list of parts. */
function contentTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let text = '';
  for (const part of content) if (isRecord(part) && typeof part.text === 'string') text += part.text;
  return text;
}

/** The answer text in a gateway body, read in the wire the gateway answered: Chat Completions. */
function answerTextOf(raw: string, stream: boolean): string {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(raw);
    } catch {
      return '';
    }
    if (!isRecord(value)) return '';
    const choices = Array.isArray(value.choices) ? value.choices : [];
    let text = '';
    for (const choice of choices) {
      if (!isRecord(choice) || !isRecord(choice.message)) continue;
      text += contentTextOf(choice.message.content);
    }
    return text;
  }
  let text = '';
  for (const frame of raw.replace(/\r\n/g, '\n').split('\n\n')) {
    const payload = frame
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
      .join('\n');
    if (payload === '' || payload === '[DONE]') continue;
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    for (const choice of choices) {
      if (!isRecord(choice) || !isRecord(choice.delta)) continue;
      if (typeof choice.delta.content === 'string') text += choice.delta.content;
    }
  }
  return text;
}

/** A body this entry already holds, as the one-chunk stream the wire writers read. */
function oneShot(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

type DialOutcome = { raw: string; used: ModelRow } | { status: number; text: string };

/**
 * Dial the plan in order and read each answer whole before relaying it. A body that does not carry
 * the prompt's token is a miss, not an answer: the next model may serve it, and the client never sees
 * a wrong answer. No clock or timeout shapes the choice, so a replay makes the same one.
 */
async function dialVerified(plan: ModelRow[], makeBody: (route: ModelRow) => Json, token: string | null): Promise<DialOutcome> {
  let lastStatus = 0;
  let lastText = '';
  for (const route of plan) {
    const request = makeBody(route);
    let upstream: Response;
    try {
      upstream = await dial(route.id, request);
    } catch (err) {
      lastStatus = 0;
      lastText = (err as Error).message;
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      lastStatus = upstream.status;
      lastText = (await upstream.text()).slice(0, 500);
      continue;
    }
    const raw = await upstream.text();
    if (token !== null && !answerTextOf(raw, request.stream === true).includes(token)) {
      lastStatus = 0;
      lastText = `the answer from ${route.id} did not carry ${token}`;
      continue;
    }
    return { raw, used: route };
  }
  return { status: lastStatus, text: lastText };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const token = tokenOf(body);
  const routes = routesFor(requirementsFor(body));
  if (routes.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const outcome = await dialVerified(attemptPlan(routes), (route) => ({ ...body, model: route.id }), token);
  if (!('raw' in outcome)) return refused(res, 'openai', outcome.status, outcome.text);
  const used = outcome.used;
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    try {
      for await (const frame of openAIStreamWithModel(oneShot(outcome.raw), used.id)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(outcome.raw);
  } catch {
    return refused(res, 'openai', 0, 'the gateway answered a body that is not JSON');
  }
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, used.id));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const token = tokenOf(body);
  const routes = routesFor(requirementsFor(body));
  if (routes.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const outcome = await dialVerified(attemptPlan(routes), (route) => anthropicToOpenAI(body, route.id), token);
  if (!('raw' in outcome)) return refused(res, 'anthropic', outcome.status, outcome.text);
  const used = outcome.used;
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    try {
      for await (const event of anthropicEventsFromOpenAI(oneShot(outcome.raw), used.id)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = JSON.parse(outcome.raw);
  } catch {
    return refused(res, 'anthropic', 0, 'the gateway answered a body that is not JSON');
  }
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, used.id));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
