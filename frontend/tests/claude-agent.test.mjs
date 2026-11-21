import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createClaudeTextMapper } from '../src/lib/claude/events.ts';

const session = 'sess-1';
const delta = (text, index = 0, parent = null) => ({
  type: 'stream_event',
  parent_tool_use_id: parent,
  session_id: session,
  event: { type: 'content_block_delta', index, delta: { type: 'text_delta', text } },
});
const assistant = (content, parent = null) => ({
  type: 'assistant',
  parent_tool_use_id: parent,
  session_id: session,
  message: { content },
});

test('text deltas stream under one item and the full message replaces them', () => {
  const map = createClaudeTextMapper();
  assert.deepEqual(map(delta('Hel')), [{ type: 'delta', itemId: `${session}:0:0`, text: 'Hel' }]);
  assert.deepEqual(map(delta('lo')), [{ type: 'delta', itemId: `${session}:0:0`, text: 'lo' }]);
  assert.deepEqual(map(assistant([{ type: 'text', text: 'Hello' }])), [
    { type: 'message', itemId: `${session}:0:0`, text: 'Hello' },
  ]);
});

test('later assistant messages get new item IDs so earlier text is kept', () => {
  const map = createClaudeTextMapper();
  map(assistant([{ type: 'text', text: 'First' }]));
  assert.deepEqual(map(delta('Sec')), [{ type: 'delta', itemId: `${session}:1:0`, text: 'Sec' }]);
  assert.deepEqual(map(assistant([{ type: 'tool_use', name: 'Read' }, { type: 'text', text: 'Second' }])), [
    { type: 'message', itemId: `${session}:1:1`, text: 'Second' },
  ]);
});

test('thinking, tool, subagent, and non-text events produce nothing', () => {
  const map = createClaudeTextMapper();
  assert.deepEqual(map({ type: 'system', session_id: session }), []);
  assert.deepEqual(map({ type: 'stream_event', parent_tool_use_id: null, session_id: session, event: { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'hmm' } } }), []);
  assert.deepEqual(map({ type: 'stream_event', parent_tool_use_id: null, session_id: session, event: { type: 'content_block_start', index: 0 } }), []);
  assert.deepEqual(map(delta('hidden', 0, 'tool-123')), []);
  assert.deepEqual(map(assistant([{ type: 'text', text: 'sub' }], 'tool-123')), []);
  assert.deepEqual(map(assistant([{ type: 'tool_use', name: 'Grep' }])), []);
  assert.deepEqual(map({ type: 'result', session_id: session }), []);
});
