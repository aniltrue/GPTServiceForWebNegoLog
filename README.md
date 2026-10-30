# GPTService

An asynchronous LLM service that decouples your application from the OpenAI API
through Redis. Clients drop a request onto a Redis queue and read the answer
back from a per-request reply queue, while the service handles concurrency,
rate-limit back-off, retries on invalid output, and usage accounting.

## Architecture

```
                 push llm:request                 OpenAI API
  ┌───────────────┐   ─────────────►  ┌───────────────┐  ───────►  ┌────────┐
  │ Your app /    │                   │   Service     │            │ OpenAI │
  │ GPTConnector  │                   │  (run.py)     │  ◄───────  └────────┘
  └───────────────┘   ◄─────────────  └───────────────┘
                 blpop                       │
            llm:response:{request_id}        ├── Messenger    (Redis I/O)
                                             ├── LLMProcessor (OpenAI calls)
                                             └── AsyncLogger  (non-blocking logs)
```

The client and the service never call each other directly — Redis is the only
contact point. This lets the service run as a standalone worker (or several of
them) independent of the processes that produce requests.

## Components

| File               | Responsibility                                                            |
| ------------------ | ------------------------------------------------------------------------- |
| `run.py`           | Entry point. Boots the `Service` and shuts it down cleanly.               |
| `Service.py`       | Main loop. Pulls requests and processes them concurrently with a semaphore. |
| `Messenger.py`     | Redis layer on the service side (listen for requests, publish responses). |
| `LLMProcess.py`    | Calls the OpenAI API with rate-limit back-off and JSON-validation retries. |
| `Models.py`        | Pydantic request/response models and `request_id` validation.             |
| `AsyncLogger.py`   | Thread-backed, non-blocking logger writing to console and `logs/`.        |
| `GPTConnector.py`  | **Client** class (`GPTServiceConnector`) for using the service from outside. |
| `test.py`          | Minimal end-to-end script that sends one request through the connector.   |

## Requirements

- Python 3.10+
- A running Redis server
- An OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The service reads configuration from `service.env` (and `.env` if present).

| Variable            | Default                       | Used by        | Description                                         |
| ------------------- | ----------------------------- | -------------- | --------------------------------------------------- |
| `REDIS_HOST`        | `localhost`                   | Messenger      | Redis host.                                         |
| `REDIS_PORT`        | `6379`                        | Messenger      | Redis port.                                         |
| `REQUEST_STREAM`    | `llm:request`                 | Messenger      | Queue the service listens on.                       |
| `RESPONSE_STREAM`   | `llm:response:{request_id}`   | Messenger      | Per-request reply queue template.                   |
| `RESPONSE_TTL`      | `3600`                        | Messenger      | Seconds an unread response stays in Redis.          |
| `MAX_CONCURRENCY`   | `10`                          | Service        | Max requests processed in parallel.                 |
| `MAX_RETRY`         | `5`                           | LLMProcessor   | Retries when the model returns invalid JSON.        |
| `RATE_MAX_RETRY`    | `10`                          | LLMProcessor   | Retries on rate-limit / transient API errors.       |
| `RATE_BACKOFF_BASE` | `1.0`                         | LLMProcessor   | Base seconds for exponential back-off.              |
| `RATE_BACKOFF_MAX`  | `10.0`                        | LLMProcessor   | Cap on back-off delay.                              |
| `OPENAI_TIMEOUT`    | `180`                         | LLMProcessor   | Seconds one OpenAI call may take.                   |
| `TIMEOUT_MAX_RETRY` | `2`                           | LLMProcessor   | Retries on an OpenAI timeout / connection error.    |
| `DEFAULT_REASONING_EFFORT` | `low`                  | LLMProcessor   | Reasoning effort when a request does not set one.   |
| `DEFAULT_TEMPERATURE` | `0.2`                       | LLMProcessor   | Temperature when a request does not set one.        |
| `OPENAI_API_KEY`    | —                             | LLMProcessor   | OpenAI API key. Put this in `.env`.                 |

> The service uses Redis `db=1`. The client (`GPTConnector.py`) must point at the
> same host, port, db and stream names, otherwise requests and responses will
> not line up.

## Running the service

```bash
python run.py
```

The service connects to Redis, starts logging to `logs/`, and begins consuming
the `llm:request` queue. Press `Ctrl+C` to stop; the messenger and logger are
closed cleanly on shutdown.

```bash
python run.py --clean
```

`--clean` first deletes the waiting requests (`llm:request`) and the unread
responses (`llm:response:*`) in Redis `db=1`. Clients still waiting on a deleted
request get a timeout. Every call logs its duration and tokens (`Usage (...)`).

## Using the service from a client

Import `GPTServiceConnector` and send a prompt. The call blocks until the
service replies (or `timeout` seconds elapse).

```python
import anyio
from GPTConnector import GPTServiceConnector

async def main():
    async with GPTServiceConnector("localhost", 6379) as gpt:
        res = await gpt.send(
            user_prompt="Return a JSON object with a `greeting` field.",
            system_prompt="You are a helpful assistant. Respond ONLY with valid JSON.",
        )
        print(res.status)    # 'Success'
        print(res.response)  # parsed JSON dict from the model

anyio.run(main)
```

`send()` parameters:

| Parameter       | Default     | Description                                              |
| --------------- | ----------- | -------------------------------------------------------- |
| `user_prompt`   | (required)  | The user message.                                        |
| `model`         | `gpt-5.4-mini` | Model name.                                           |
| `system_prompt` | `None`      | Optional system message.                                 |
| `table_code`    | `llm`       | Prefix of the generated `request_id`; must match the service validation. |
| `reasoning_effort` | `None`   | `none`/`minimal`/`low`/`medium`/`high`/`xhigh`; service default if omitted. |
| `temperature`   | `None`      | 0-2; service default if omitted.                         |

If the model rejects `temperature` or `reasoning_effort` (some reasoning models
accept only the default temperature), the service logs a warning and retries the
request without that parameter.

On a service-side error or a timeout, `send()` raises `GPTServiceError`.

## Quick test

With Redis up and the service running (`python run.py`), in a separate terminal:

```bash
python test.py
```

It sends one request through the connector and prints the status and response.

## Message format

**Request** (pushed to `llm:request`):

```json
{
  "RequestId": "llm-2f1c8e7a-....",
  "Model": "gpt-5.4-mini",
  "UserPrompt": "...",
  "SystemPrompt": "...",
  "ReasoningEffort": "low",
  "Temperature": 0.2
}
```

**Response** (pushed to `llm:response:{request_id}`):

```json
{
  "Status": "Success",
  "ErrorMessage": null,
  "Response": { "...": "model output as a JSON object" }
}
```

The model is always asked for a JSON object (`response_format=json_object`), so
`Response` is the parsed JSON returned by the model. On failure, `Status` is
`"Error"` and `ErrorMessage` describes what went wrong.
