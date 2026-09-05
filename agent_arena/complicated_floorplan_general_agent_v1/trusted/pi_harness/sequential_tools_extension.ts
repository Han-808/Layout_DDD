/**
 * Benchmark-owned Pi tool boundary.
 *
 * Pi's stock tools may execute multiple tool calls in parallel and its stock
 * bash tool inherits the Pi process environment.  Official SIEVE episodes
 * require one-at-a-time tool execution, and a shell subprocess must never
 * inherit the short-lived model-gateway capability used by Pi itself.
 */
import { spawn } from "node:child_process";
import { readSync, writeSync } from "node:fs";
import type {
  BashOperations,
  ExtensionAPI,
  ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import {
  createBashToolDefinition,
  createEditToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
} from "@earendil-works/pi-coding-agent";

const MODEL_GATEWAY_CAPABILITY = "ARENA_MODEL_GATEWAY_TOKEN";
const BASH_REGISTRY_FD = "SIEVE_BASH_REGISTRY_FD";
const BASH_REGISTRY_ACK_FD = "SIEVE_BASH_REGISTRY_ACK_FD";
const BASH_REGISTRY_NONCE = "SIEVE_BASH_REGISTRY_NONCE";
const MAX_TIMEOUT_MS = 2_147_483_647;
const POST_EXIT_OUTPUT_GRACE_MS = 100;
const TERMINATE_GRACE_MS = 250;

function sequential<T extends ToolDefinition>(definition: T): T {
  return { ...definition, executionMode: "sequential" } as T;
}

const MAX_ACK_BYTES = 4096;

type Registry = Readonly<{ fd: number; ackFd: number; nonce: string }>;

function loadRegistry(): Registry {
  const rawFd = process.env[BASH_REGISTRY_FD];
  const rawAckFd = process.env[BASH_REGISTRY_ACK_FD];
  const nonce = process.env[BASH_REGISTRY_NONCE];
  if (
    !rawFd ||
    !rawAckFd ||
    !/^[0-9]+$/.test(rawFd) ||
    !/^[0-9]+$/.test(rawAckFd) ||
    !nonce
  ) {
    throw new Error("SIEVE Bash supervisor channel is unavailable");
  }
  const fd = Number(rawFd);
  const ackFd = Number(rawAckFd);
  if (
    !Number.isSafeInteger(fd) ||
    !Number.isSafeInteger(ackFd) ||
    fd < 3 ||
    ackFd < 3 ||
    fd === ackFd ||
    nonce.length !== 64 ||
    !/^[0-9a-f]+$/.test(nonce)
  ) {
    throw new Error("SIEVE Bash supervisor channel is malformed");
  }
  return { fd, ackFd, nonce };
}

function emitProcessEvent(
  registry: Registry,
  event: "start" | "exit",
  pid: number,
): void {
  const record = JSON.stringify({
    schema_version: "sieve_bash_process_event_v1",
    nonce: registry.nonce,
    event,
    pid,
  });
  // One write, well below PIPE_BUF, keeps records atomic. The shell child never
  // inherits this descriptor or its nonce.
  writeSync(registry.fd, `${record}\n`, undefined, "utf8");
}

function waitForStartAcknowledgement(registry: Registry, pid: number): void {
  const bytes: number[] = [];
  const one = Buffer.allocUnsafe(1);
  while (bytes.length < MAX_ACK_BYTES) {
    const count = readSync(registry.ackFd, one, 0, 1, null);
    if (count === 0) {
      throw new Error("SIEVE Bash supervisor acknowledgement channel closed");
    }
    if (one[0] === 0x0a) break;
    bytes.push(one[0]);
  }
  if (bytes.length === MAX_ACK_BYTES) {
    throw new Error("SIEVE Bash supervisor acknowledgement is oversized");
  }
  let acknowledgement: unknown;
  try {
    acknowledgement = JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    throw new Error("SIEVE Bash supervisor acknowledgement is malformed");
  }
  if (
    typeof acknowledgement !== "object" ||
    acknowledgement === null ||
    Array.isArray(acknowledgement)
  ) {
    throw new Error("SIEVE Bash supervisor acknowledgement is invalid");
  }
  const record = acknowledgement as Record<string, unknown>;
  if (
    Object.keys(record).sort().join(",") !== "nonce,pid,schema_version" ||
    record.schema_version !== "sieve_bash_process_ack_v1" ||
    record.nonce !== registry.nonce ||
    record.pid !== pid
  ) {
    throw new Error("SIEVE Bash supervisor acknowledgement did not match");
  }
}

/**
 * Pi 0.85's stock local Bash backend starts every command with
 * `detached: true`. That creates a new session outside the episode PGID, so an
 * outer timeout cannot prove that the command's descendants are gone. This
 * backend is deliberately owned by the benchmark instead:
 *
 * 1. the shell remains in Pi's episode process group (`detached: false`);
 * 2. it blocks on stdin until the trusted outer supervisor has received a
 *    start record; and
 * 3. an exit record lets the supervisor reject a command that left background
 */
function createSupervisedBashOperations(registry: Registry): BashOperations {
  return {
    exec: (command, cwd, { onData, signal, timeout, env }) =>
      new Promise((resolve, reject) => {
        if (signal?.aborted) {
          reject(new Error("aborted"));
          return;
        }
        const timeoutMs = timeout === undefined ? undefined : timeout * 1000;
        if (
          timeoutMs !== undefined &&
          (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > MAX_TIMEOUT_MS)
        ) {
          reject(new Error("Invalid timeout: must be finite, positive, and bounded"));
          return;
        }

        // `-s` is intentional: no model-controlled command byte is delivered
        // until the process has been registered with the host supervisor.
        const childEnvironment = { ...env };
        delete childEnvironment[MODEL_GATEWAY_CAPABILITY];
        delete childEnvironment[BASH_REGISTRY_FD];
        delete childEnvironment[BASH_REGISTRY_ACK_FD];
        delete childEnvironment[BASH_REGISTRY_NONCE];
        const child = spawn("/bin/bash", ["-s"], {
          cwd,
          detached: false,
          env: childEnvironment,
          stdio: ["pipe", "pipe", "pipe"],
          windowsHide: true,
        });
        let settled = false;
        let exited = false;
        let exitCode: number | null = null;
        let stopReason: "aborted" | "timeout" | null = null;
        let timeoutHandle: NodeJS.Timeout | undefined;
        let killHandle: NodeJS.Timeout | undefined;
        let finishHandle: NodeJS.Timeout | undefined;
        let registered = false;

        const cleanup = () => {
          if (timeoutHandle) clearTimeout(timeoutHandle);
          if (killHandle) clearTimeout(killHandle);
          if (finishHandle) clearTimeout(finishHandle);
          signal?.removeEventListener("abort", onAbort);
          child.stdout.removeListener("data", onData);
          child.stderr.removeListener("data", onData);
          child.removeListener("error", onError);
          child.removeListener("exit", onExit);
          child.stdout.destroy();
          child.stderr.destroy();
        };

        const finish = () => {
          if (settled) return;
          settled = true;
          cleanup();
          if (stopReason === "aborted") {
            reject(new Error("aborted"));
          } else if (stopReason === "timeout") {
            reject(new Error(`timeout:${timeout}`));
          } else {
            resolve({ exitCode });
          }
        };

        const requestStop = (reason: "aborted" | "timeout") => {
          if (settled || exited || stopReason) return;
          stopReason = reason;
          try {
            child.kill("SIGTERM");
          } catch {
            // The child may have crossed the exit boundary concurrently.
          }
          killHandle = setTimeout(() => {
            try {
              child.kill("SIGKILL");
            } catch {
              // Already gone.
            }
          }, TERMINATE_GRACE_MS);
        };

        const onAbort = () => requestStop("aborted");
        const onError = (error: Error) => {
          if (settled) return;
          settled = true;
          cleanup();
          reject(error);
        };
        const onExit = (code: number | null) => {
          if (settled) return;
          exited = true;
          exitCode = code;
          try {
            if (registered && child.pid) emitProcessEvent(registry, "exit", child.pid);
          } catch (error) {
            settled = true;
            cleanup();
            reject(error);
            return;
          }
          // Preserve the stock Pi backend's short post-exit output grace, but
          // never wait indefinitely for pipes held by an escaped background job.
          finishHandle = setTimeout(finish, POST_EXIT_OUTPUT_GRACE_MS);
        };

        child.stdout.on("data", onData);
        child.stderr.on("data", onData);
        child.once("error", onError);
        child.once("exit", onExit);
        signal?.addEventListener("abort", onAbort, { once: true });

        if (!child.pid) {
          requestStop("aborted");
          reject(new Error("SIEVE Bash child has no process identity"));
          return;
        }
        try {
          emitProcessEvent(registry, "start", child.pid);
          waitForStartAcknowledgement(registry, child.pid);
          registered = true;
          child.stdin.on("error", () => {});
          child.stdin.end(`${command}\n`);
        } catch (error) {
          // No model-controlled command byte has been delivered.  Kill the
          // empty shell immediately; the host also owns it by the start
          // record and independently proves tree cleanup.
          try {
            child.kill("SIGKILL");
          } catch {
            // The child may have crossed the exit boundary concurrently.
          }
          settled = true;
          cleanup();
          reject(error);
          return;
        }
        if (timeoutMs !== undefined) {
          timeoutHandle = setTimeout(() => requestStop("timeout"), timeoutMs);
        }
      }),
  };
}

export default function registerSieveTools(pi: ExtensionAPI) {
  const cwd = process.cwd();
  const registry = loadRegistry();

  pi.registerTool(sequential(createReadToolDefinition(cwd)));
  pi.registerTool(sequential(createWriteToolDefinition(cwd)));
  pi.registerTool(sequential(createEditToolDefinition(cwd)));
  pi.registerTool(
    sequential(
      createBashToolDefinition(cwd, {
        operations: createSupervisedBashOperations(registry),
        exposeSessionEnvironment: false,
        commandPrefix: `unset ${MODEL_GATEWAY_CAPABILITY}`,
        spawnHook: ({ command, cwd: commandCwd, env }) => {
          const childEnvironment = { ...env };
          delete childEnvironment[MODEL_GATEWAY_CAPABILITY];
          delete childEnvironment[BASH_REGISTRY_FD];
          delete childEnvironment[BASH_REGISTRY_ACK_FD];
          delete childEnvironment[BASH_REGISTRY_NONCE];
          return {
            command,
            cwd: commandCwd,
            env: childEnvironment,
          };
        },
      }),
    ),
  );
}
